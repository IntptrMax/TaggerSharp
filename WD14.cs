using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TaggerSharp
{
    public class WD14
    {
        /// <summary>
        /// 配置类，用于存储模型和标签的相关配置参数。
        /// </summary>
        public class Config
        {
            /// <summary>
            /// 资源文件的路径。
            /// </summary>
            public string AssetsPath { get; set; } = @"..\..\..\Assets\";

            /// <summary>
            /// 张量的数据类型，默认为 Float16。
            /// </summary>
            public ScalarType ScalarType { get; set; } = ScalarType.Float16;

            /// <summary>
            /// 设备类型，默认为 CUDA。
            /// </summary>
            public DeviceType DeviceType { get; set; } = DeviceType.CUDA;

            /// <summary>
            /// 预测标签的阈值，默认为 0.3。
            /// </summary>
            public float Threshold { get; set; } = 0.3f;

            /// <summary>
            /// 标签文本文件的文件名。
            /// </summary>
            public string TagTextFile { get; set; } = "tags.csv";

            /// <summary>
            /// TorchScript 模型文件的文件名。
            /// </summary>
            public string TorchScriptFile { get; set; } = "vit_fp16.torchscript";

            /// <summary>
            /// 模型使用的图片尺寸
            /// 默认 448，基本不变，除非模型有特殊要求
            /// </summary>
            public int ImageSize { get; set; } = 448;

            /// <summary>
            /// 一次性让Gpu处理的图片数量
            /// </summary>
            public int BatchSize { get; set; } = 32;

            /// <summary>
            /// 数据的处理线程
            /// 如果是 -1 则在初始化时，读取当前cpu的核心数
            /// </summary>
            public int NumWorker { get; set; } = -1;

            /// <summary>
            /// 是否使用预加载图片，默认为false
            /// </summary>
            public bool UsePreload { get; set; } = false;

            /// <summary>
            /// 如果使用预加载图片，缓存图片的数量
            /// </summary>
            public int ImageCacheSize { get; set; } = 512;
        }

        Config _config;
        List<TagItem> _tags;
        jit.ScriptModule _model;

        public WD14(Config config)
        {
            _config = config;
            if (_config.NumWorker == -1)
                _config.NumWorker = Environment.ProcessorCount;

            _tags = LoadTags();

            // 设置用于计算的线程数，一般设为物理核心数
            torch.set_num_threads(_config.NumWorker);
            // 设置并发操作的线程数，也可设为物理核心数
            torch.set_num_interop_threads(_config.NumWorker);

            // 设置环境变量
            Environment.SetEnvironmentVariable("OMP_NUM_THREADS", _config.NumWorker.ToString());
            Environment.SetEnvironmentVariable("MKL_NUM_THREADS", _config.NumWorker.ToString());

            var modelFileName = Path.Combine(_config.AssetsPath, _config.TorchScriptFile);
            _model = jit.load(modelFileName, config.DeviceType).to(config.ScalarType);
        }

        /// <summary>
        /// 获得一张图的标签数据
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public ImageTagResult GetImageTag(string fileName)
        {
            return GetImageTags(new[] { fileName }).First();
        }

        public IEnumerable<ImageTagResult> GetImageTagByFolder(string folder)
        {
            string[] imagesFileNames = Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories).Where(file =>
            {
                // 获取文件扩展名
                string extension = Path.GetExtension(file).ToLower();
                // 筛选出指定格式的图像文件
                return extension == ".jpg" || extension == ".png" || extension == ".bmp" || extension == ".webp";
            }).ToArray();

            return GetImageTags(imagesFileNames);
        }

        /// <summary>
        /// 获得一张图的标签数据
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public IEnumerable<ImageTagResult> GetImageTags(string[] fileNames)
        {
            ImageDataSet dataset = new ImageDataSet(fileNames,
                size: _config.ImageSize,
                cacheSize: _config.ImageCacheSize,
                usePreload: _config.UsePreload,
                deviceType: _config.DeviceType,
                scalarType: _config.ScalarType);

            DataLoader dataLoader = new DataLoader(dataset, _config.BatchSize,
                num_worker: _config.NumWorker,
                device: new Device(_config.DeviceType),
                shuffle: false);

            Stopwatch sw = Stopwatch.StartNew();

            int step = 0;
            using (torch.no_grad())
            {
                foreach (var imageData in dataLoader)
                {
                    step++;
                    Console.WriteLine("Process: {0}/{1}......", step, dataLoader.Count);
                    Tensor images = imageData["image"];
                    Tensor indexs = imageData["index"];

                    Tensor result = (Tensor)_model.forward(images);

                    // 对 result 张量应用 sigmoid 函数，然后选择从第5列（索引4）开始的所有列
                    var sigmoid_result = result.sigmoid()[TensorIndex.Ellipsis, 4..];

                    // 对 result 张量按行（dim: 1）进行排序，并返回排序后的索引，按降序排序
                    var sortedIndices = sigmoid_result.argsort(dim: 1, descending: true);

                    // 创建一个布尔掩码，表示 result 张量中大于 threshold 的元素
                    var mask = sigmoid_result > _config.Threshold; ;

                    // 对 mask 张量按行求和，得到每行中大于 threshold 的元素数量，并保持维度
                    Tensor countTensor = mask.sum(1, true);

                    // 将 countTensor 转换为 long 类型的数组，表示每行中大于 threshold 的元素数量
                    long[] counts = countTensor.data<long>().ToArray();

                    for (int i = 0; i < counts.Length; i++)
                    {
                        StringBuilder stringBuilder = new StringBuilder();
                        string name = Path.GetFileNameWithoutExtension(dataset.GetFileName(indexs[i].ToInt64()));
                        long[] sortedIndex = sortedIndices[i].data<long>().ToArray();
                        float[] tagProbabilities = sigmoid_result[i].to(torch.float32).data<float>().ToArray();
                        ImageTagResult ret = new ImageTagResult
                        {
                            FileName = name,
                            Tags = new List<ImageTagResultItem>()
                        };

                        for (int j = 0; j < counts[i]; j++)
                        {
                            var tagIndex = (int)(sortedIndex[j]);
                            var probability = tagProbabilities[tagIndex];

                            ret.Tags.Add(new ImageTagResultItem
                            {
                                Info = _tags[tagIndex],
                                Probability = probability
                            });
                        }

                        // 返回推理结果
                        yield return ret;
                    }
                }
            }

            Console.WriteLine("All done. Tag {0} images, using {1} seconds.", dataset.Count, sw.Elapsed.TotalSeconds);
        }


        public List<TagItem> LoadTags()
        {
            List<TagItem> tags = new List<TagItem>();
            var fileName = Path.Combine(_config.AssetsPath, _config.TagTextFile);

            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException("tags2.csv not found:" + fileName);
            }

            // 第一行是标题，格式是： tag_id,name,category,count,zh
            // 前 4 个标签是不能用的，也需要跳过

            var lines = File.ReadAllLines(fileName).Skip(5);
            foreach (string line in lines)
            {
                string[] items = line.Split(',');
                tags.Add(new TagItem
                {
                    TagId = int.Parse(items[0]),
                    Name = items[1],
                    Category = items[2],
                    Count = int.Parse(items[3]),
                    Zh = items[4]
                });
            }
            return tags;
        }
    }

    public class ImageTagResult
    {
        /// <summary>
        ///文件名。
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        ///标签列表。
        /// </summary>
        public List<ImageTagResultItem> Tags { get; set; }
    }

    public class ImageTagResultItem
    {
        /// <summary>
        ///标签信息。
        /// </summary>
        public TagItem Info { get; set; }

        /// <summary>
        ///标签的概率。
        /// </summary>
        public float Probability { get; set; }
    }

    public class TagItem
    {
        /// <summary>
        /// 标签ID（是指原始数据里，打标网站的标签ID）
        /// 不是推理后的出现的标签id。
        /// 所以实际上这个id没有什么用
        /// </summary>
        public int TagId { get; set; }

        /// <summary>
        /// 标签名称。
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// 标签类别。
        /// </summary>
        public string Category { get; set; }

        /// <summary>
        /// 项目之前训练时有的数量
        /// </summary>
        public int Count { get; set; }

        /// <summary>
        /// 标签的中文名称。
        /// 不一定所有的标签都存在
        /// </summary>
        public string Zh { get; set; }
    }
}
