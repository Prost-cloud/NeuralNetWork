using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using NeuralNetwork.Network;
using System.IO;
using Microsoft.Win32;
using System.Threading.Tasks;
using System.Drawing;
//using System.Drawing.Common;

namespace NeuralNetWorkView
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NeuralNetwork.Network.NeuralNetwork neuralNetwork
            = new NeuralNetwork.Network.NeuralNetwork(0.0001d, 784, 512, 128, 32, 10);



        System.Windows.Point currentPoint = new System.Windows.Point();
        public MainWindow()
        {
            InitializeComponent();
            LoadComponenet();
        }
        private void LoadComponenet()
        {
            ScaleTransform transform = new ScaleTransform();

            transform.ScaleX = 12;//1
            transform.ScaleY = 12;//1

            paintSurface.RenderTransform = transform;

            WriteableBitmap writeableBitmap = new WriteableBitmap(28, 28, 96, 96, PixelFormats.Rgb24, null);
            //WriteableBitmap writeableBitmap = new WriteableBitmap(28, 28, 256, 256, PixelFormats.Rgb24, null);

            System.Windows.Controls.Image image = new System.Windows.Controls.Image();
            image.Stretch = Stretch.None;
            image.Margin = new Thickness(0);
            Grid1.Children.Add(image);

        }


        private void Canvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ButtonState == MouseButtonState.Pressed)
                currentPoint = e.GetPosition(paintSurface);
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Line line = new Line();

                line.Stroke = System.Windows.SystemColors.ActiveCaptionTextBrush;
                line.X1 = currentPoint.X;
                line.Y1 = currentPoint.Y;
                line.X2 = e.GetPosition(paintSurface).X;
                line.Y2 = e.GetPosition(paintSurface).Y;

                line.StrokeThickness = 3;

                //line.c
                //line.RenderSize = new System.Windows.Size(8, 8);

                currentPoint = e.GetPosition(paintSurface);

                paintSurface.Children.Add(line);

                //int[,] rgbArr = new int[144,144];

                //var image = new BitmapImage();
                //image.BeginInit();

                //image.EndInit();

                //var bitMap = paintSurface.Children.

                //paintSurface.
            }
        }

        private void Button_Clear_Click(object sender, RoutedEventArgs e)
        {

            paintSurface.Children.Clear();
        }

        private void Button_WhatIsIt_Click(object sender, RoutedEventArgs e)
        {
            var pSource = PresentationSource.FromVisual(paintSurface);
            Matrix matrix = pSource.CompositionTarget.TransformToDevice;

            double dpiX = matrix.M11 * 96;
            double dpiY = matrix.M22 * 96;
            //double dpiX = matrix.M11 * 256;
            //double dpiY = matrix.M22 * 256;

            var elementBitmap = new RenderTargetBitmap(28, 28, dpiX, dpiY, PixelFormats.Default);

            var drawingVisual = new DrawingVisual();

            using (DrawingContext drawingContext = drawingVisual.RenderOpen())
            {
                var visualBrush = new VisualBrush(paintSurface);
                drawingContext.DrawRectangle(visualBrush,
                    null,
                    new Rect(new System.Windows.Point(0, 0), new System.Windows.Size(28, 28)));
            }
            elementBitmap.Render(drawingVisual);

            int[] dArray = new int[28 * 28];

            int nStride = (elementBitmap.PixelWidth * elementBitmap.Format.BitsPerPixel + 7) / 8;

            elementBitmap.CopyPixels(dArray, nStride, 0);

            for (int i = 0; i < dArray.Length; i++)
            {
                dArray[i] = 0xffffff - dArray[i];
            }

            double[] input = new double[28 * 28];
            int[] inputToImg = new int[28 * 28];
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                {
                    input[i + j * 28] = (((dArray[i + j * 28]) & 0xFF) / 255d);
                    inputToImg[i + j * 28] = (dArray[i + j * 28]) & 0xFF;
                    // if (dArray[i + j * 28] > 0x90)
                    //      input[i + j * 28] = 1;
                }
            using (Bitmap bitmap = new Bitmap(28, 28))
            {
                for (int i = 0; i < 28; i++)
                    for (int j = 0; j < 28; j++)
                        bitmap.SetPixel(i, j, System.Drawing.Color.FromArgb(inputToImg[i + j * 28]));

                bitmap.Save(@"F:\Project\imageCheck.bmp", System.Drawing.Imaging.ImageFormat.Bmp);
            }

                GetPredict(input);
        }

        private async void GetPredict(double[] input)
        {
            double[] outputs = await neuralNetwork.FeedForward(input);
            //encoder.Frames.Add(BitmapFrame.Create(elementBitmap));
            int result = -1;
            double probability = -1000d;
            for (int i = 0; i < 10; i++)
            {
                if (probability < outputs[i])
                {
                    result = i;
                    probability = outputs[i];
                }
            }
            ResultText.Text = $"Result {result}. Probability {probability}";
        }

        private async void Button_Train_Click(object sender, RoutedEventArgs e)
        {
            Button_Train_ClickAsync(sender, e);
        }

        private async Task Button_Train_ClickAsync(object sender, RoutedEventArgs e)
        {
            int samples = 60000;
            double[][] inputs = new double[samples][];
            int[] digits = new int[samples];

            {
                Bitmap[] images = new Bitmap[samples];

                string[] files = Directory.GetFiles("F:\\Project\\NeuralNetWorkView\\NeuralNetWorkView\\bin\\Debug\\netcoreapp3.1\\train");

                for (int i = 0; i < samples; i++)
                {
                    images[i] = new Bitmap(files[i]);
                    digits[i] = int.Parse(files[i][^5].ToString());
                }

                for (int i = 0; i < samples; i++)
                {
                    inputs[i] = new double[28 * 28];
                    for (int x = 0; x < 28; x++)
                        for (int y = 0; y < 28; y++)
                        {
                            inputs[i][x + y * 28] = ((images[i].GetPixel(x, y).ToArgb()) & 0xFF) / 255d;
                        }
                }
            }
            int epoch = 200;

            Random random = new Random();

            for (int i = 0; i < epoch; i++)
            {
                int right = 0;
                double errorSum = 0;
                int batchSize = 100;

                for (int j = 0; j <= batchSize; j++)
                {
                    int imgIndex = random.Next(0, samples);
                    double[] targets = new double[10];
                    int digit = digits[imgIndex];
                    targets[digit] = 1;

                    double[] outputs = await neuralNetwork.FeedForward(inputs[imgIndex]);
                    int maxDigit = 0;

                    double maxDigitWeight = -1;

                    for (int k = 0; k < 10; k++)
                    {
                        if (outputs[k] > maxDigitWeight)
                        {
                            maxDigitWeight = outputs[k];
                            maxDigit = k;
                        }
                    }

                    if (digit == maxDigit) right++;

                    for (int k = 0; k < 10; k++)
                    {
                        errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                    }
                    neuralNetwork.Backpropagation(targets);

                }
                TextBox.Items.Add($"epoch: {i}. Correct {right}. error: {errorSum}");
            }
            return;

        }
        private void Button_SaveWeights_Click(object sender, RoutedEventArgs e)
        {
            string weights = neuralNetwork.GetWeights();
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.Filter = "txt files (*.txt)|*.txt";
            saveFileDialog.FilterIndex = 2;
            saveFileDialog.RestoreDirectory = true;

            if (saveFileDialog.ShowDialog() == true)
            {
                File.WriteAllText(saveFileDialog.FileName, weights);
            }

        }
        private void Button_LoadWeights_Click(object sender, RoutedEventArgs e)
        {
            var pathToFile = new OpenFileDialog();
            pathToFile.Filter = "txt files (*.txt)|*.txt";
            pathToFile.RestoreDirectory = true;

            string weights = string.Empty;

            if (pathToFile.ShowDialog() == true)
            {
                weights = File.ReadAllLines(pathToFile.FileName)[0];
            }

            if (weights != string.Empty)
            {
                neuralNetwork.SetWeightsFromString(weights);
            }
        }

        private void Button_TestCase_Click(object sender, RoutedEventArgs e)
        {
            var pathToFile = new OpenFileDialog();
            pathToFile.Filter = "png files (*.png)|*.png";
            pathToFile.RestoreDirectory = true;

            var input = new double[28 * 28];

            if (pathToFile.ShowDialog() == true)
            {
                var png = new Bitmap(pathToFile.FileName);

                for (int x = 0; x < 28; x++)
                    for (int y = 0; y < 28; y++)
                    {
                        input[x + y * 28] = ((png.GetPixel(x, y).ToArgb()) & 0xFF) / 255d;
                    }
            }

            GetPredict(input);

        }
    }
}
