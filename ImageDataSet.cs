using TorchSharp;
using static TorchSharp.torch;

namespace TaggerSharp
{
	internal class ImageDataSet : utils.data.Dataset
	{
		private List<string> imageFiles = new List<string>();
		private Device device;
		private ScalarType scalarType;
		private int size;
		private int cacheSize;
		private Dictionary<int, Tensor> cacheImages = new Dictionary<int, Tensor>();
		private int currentIndex = 0;
		private bool usePreload = false;

        /// <summary>
        /// 初始化 ImageDataSet 类的实例，根据指定的参数加载图像文件并进行预处理。
        /// </summary>
        /// <param name="rootPath">图像文件的根目录路径。</param>
        /// <param name="size">调整图像大小的目标尺寸，默认为 448。</param>
        /// <param name="cacheSize">缓存大小，默认为 800。</param>
        /// <param name="usePreload">是否使用预加载，默认为 false。</param>
        /// <param name="deviceType">设备类型，默认为 CUDA。</param>
        /// <param name="scalarType">张量的数据类型，默认为 Float16。</param>
        public ImageDataSet(string rootPath, int size = 448, int cacheSize = 800, bool usePreload = false, DeviceType deviceType = DeviceType.CUDA, ScalarType scalarType = ScalarType.Float16)
        {
            string[] imagesFileNames = Directory.GetFiles(rootPath, "*.*", SearchOption.AllDirectories).Where(file =>
            {
                // 获取文件扩展名
                string extension = Path.GetExtension(file).ToLower();
                // 筛选出指定格式的图像文件
                return extension == ".jpg" || extension == ".png" || extension == ".bmp" || extension == ".webp";
            }).ToArray();

            // 调用另一个构造函数来初始化
            Init(imagesFileNames, size, cacheSize, usePreload, deviceType, scalarType);
        }

        /// <summary>
        /// 初始化 ImageDataSet 类的实例，根据指定的参数加载图像文件并进行预处理。
        /// </summary>
        /// <param name="imagesFileNames">图片文件列表。</param>
        /// <param name="size">调整图像大小的目标尺寸，默认为 448。</param>
        /// <param name="cacheSize">缓存大小，默认为 800。</param>
        /// <param name="usePreload">是否使用预加载，默认为 false。</param>
        /// <param name="deviceType">设备类型，默认为 CUDA。</param>
        /// <param name="scalarType">张量的数据类型，默认为 Float16。</param>
        public ImageDataSet(string[] imagesFileNames, int size = 448, int cacheSize = 800, bool usePreload = false, DeviceType deviceType = DeviceType.CUDA, ScalarType scalarType = ScalarType.Float16)
        {
            Init(imagesFileNames, size, cacheSize, usePreload, deviceType, scalarType);
        }

        private void Init(string[] imagesFileNames, int size, int cacheSize, bool usePreload, DeviceType deviceType, ScalarType scalarType)
        {
            // 将所有图像文件路径添加到列表中
            imageFiles.AddRange(imagesFileNames);

            // 设置是否使用预加载
            this.usePreload = usePreload;

            // 设置默认的图像读取器
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

            // 初始化设备和数据类型
            this.device = new Device(deviceType);
            this.scalarType = scalarType;

            // 设置图像大小和缓存大小
            this.size = size;
            this.cacheSize = cacheSize;

            if (usePreload)
            {
                // 如果使用预加载，启动一个后台任务来预加载图像
                Task.Run(() =>
                {
                    while (true)
                    {
                        PreLoadImages();
                    }
                });

                // 等待缓存加载到一半或全部图像加载完成
                while (cacheImages.Count < cacheSize / 2 && cacheImages.Count < Count)
                {
                    Thread.Sleep(10);
                }
            }
        }

        /// <summary>
        /// 预加载图像到缓存中，以提高加载效率。
        /// </summary>
        private void PreLoadImages()
        {
            // 计算缓存大小的一半
            int halfSize = cacheSize / 2;

            // 遍历当前索引附近的图像索引
            for (int i = currentIndex - halfSize; i < currentIndex + halfSize; i++)
            {
                // 检查索引是否在有效范围内
                if (i > -1 && i < Count)
                {
                    // 如果缓存中不包含当前索引的图像
                    if (!cacheImages.ContainsKey(i))
                    {
                        // 读取图像并添加到缓存中
                        Tensor Img = torchvision.io.read_image(imageFiles[i]);
                        cacheImages.Add(i, Img);
                    }
                }
            }

            // 找出超出缓存范围的图像索引
            List<int> list = cacheImages.Keys.Where(x => x > currentIndex + halfSize || x < currentIndex - halfSize).ToList();

            // 从缓存中移除超出范围的图像
            foreach (int i in list)
            {
                cacheImages.Remove(i);
            }

            Thread.Sleep(2);
        }

		public override long Count => imageFiles.Count;

		public string GetFileName(long index)
		{
			return imageFiles[(int)index];
		}

        /// <summary>
        /// 根据提供的索引获取对应的图像张量和索引。
        /// </summary>
        /// <param name="index">图像的索引。</param>
        /// <returns>包含图像张量和索引的字典。</returns>
        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            // 更新当前索引
            currentIndex = (int)index;

            // 初始化输出字典
            Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();

            // 定义一个空张量用于存储图像
            Tensor Img = torch.zeros(0);

            if (usePreload)
            {
                // 如果使用预加载，从缓存中获取图像并进行类型转换和归一化
                Img = cacheImages.GetValueOrDefault(currentIndex).to(scalarType, device) / 255.0f;
            }
            else
            {
                // 如果不使用预加载，读取图像文件并进行类型转换和归一化
                Img = torchvision.io.read_image(imageFiles[(int)index]).to(scalarType, device) / 255.0f;
            }

            // 调整图像尺寸
            Img = Letterbox(Img, size, size);

            // 将图像通道从RGB转换为BGR
            Tensor r = Img[0].unsqueeze(0);
            Tensor g = Img[1].unsqueeze(0);
            Tensor b = Img[2].unsqueeze(0);
            Img = torch.cat(new Tensor[] { b, g, r });

            // 将图像张量和索引添加到输出字典中
            outputs.Add("image", Img);
            outputs.Add("index", torch.tensor(index));

            // 返回包含图像和索引的字典
            return outputs;
        }

        /// <summary>
        /// 调整图像大小并填充以适应目标宽度和高度，同时保持图像的纵横比。
        /// </summary>
        /// <param name="image">输入的图像张量。</param>
        /// <param name="targetWidth">目标宽度。</param>
        /// <param name="targetHeight">目标高度。</param>
        /// <returns>调整大小并填充后的图像张量。</returns>
        private Tensor Letterbox(Tensor image, int targetWidth, int targetHeight)
        {
            // 获取原始图像的宽度和高度
            int originalWidth = (int)image.shape[2];
            int originalHeight = (int)image.shape[1];

            // 计算缩放比例，以保持图像的纵横比
            float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

            // 计算缩放后的宽度和高度
            int scaledWidth = (int)(originalWidth * scale);
            int scaledHeight = (int)(originalHeight * scale);

            // 计算填充的左边和顶部的大小
            int padLeft = (targetWidth - scaledWidth) / 2;
            int padTop = (targetHeight - scaledHeight) / 2;

            // 调整图像大小
            Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

            // 创建一个填充后的图像张量，并将缩放后的图像复制到填充后的图像张量中
            Tensor paddedImage = torch.ones([3, targetHeight, targetWidth], image.dtype, image.device);
            paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

            return paddedImage;
        }

    }
}
