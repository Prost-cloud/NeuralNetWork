using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading.Tasks;

namespace NeuralNetwork.Network
{
    public class NeuralNetwork
    {
        private double _learningRate;
        private Layer[] _layers;
        private readonly Func<double, double> _activation = (x) => 1 / (1 + Math.Exp(-x));
        private readonly Func<double, double> _derivative = (x) => x * (1 - x);

        public NeuralNetwork(double learningRate, params int[] sizes)
        {
            Random random = new Random();
            _learningRate = learningRate;
            _layers = new Layer[sizes.Length];
            for (int i = 0; i < sizes.Length; i++)
            {
                int nextSize = 0;
                if (i < sizes.Length - 1)
                    nextSize = sizes[i + 1];

                _layers[i] = new Layer(sizes[i], nextSize);

                for (int j = 0; j < sizes[i]; j++)
                {
                    _layers[i].Biases[j] = random.NextDouble() * 2d - 1d;

                    for (int k = 0; k < nextSize; k++)
                    {
                        _layers[i].Weights[j, k] = random.NextDouble() * 2d - 1d;
                    }
                }
            }
        }
      
        public NeuralNetwork(double learningRate, Func<double, double> activation, Func<double, double> derivative, params int[] sizes) : this(learningRate, sizes)
        {
            _activation = activation;
            _derivative = derivative;
        }

        public string GetWeights()
        {
            StringBuilder sb = new StringBuilder();

            for (int k = 0; k < _layers.Length - 1; k++)
            {
                for (int i = 0; i < _layers[k].Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < _layers[k].Weights.GetLength(1); j++)
                    {
                        sb.Append(_layers[k].Weights[i, j]);
                        if (j != _layers[k].Weights.GetLength(1) - 1)
                            sb.Append(".");
                    }
                    if (i != _layers[k].Weights.GetLength(0) - 1)
                        sb.Append(";");
                }
                if (k != _layers.Length - 1)
                    sb.Append(":");
            }
            return sb.ToString();
        }

        public void SetWeightsFromString(string weights)
        {
            var strAllLayers = weights.Split(":");

            for (int k = 0; k < strAllLayers.Length - 1; k++)
            {
                var layer = _layers[k];

                var strLayer = strAllLayers[k].Split(";");
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                {
                    var strWeights = strLayer[i].Split(".");

                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                    {
                        if (strWeights.Length != layer.Weights.GetLength(1))
                        {
                            throw new ArgumentException("That file does not fit to that neural network");
                        }
                        layer.Weights[i, j] = double.Parse(strWeights[j]);
                    }
                }
            }
        }

        public async Task<double[]> FeedForward(double[] inputs)
        {
            Array.Copy(inputs, 0, _layers[0].Neurons, 0, inputs.Length);


            for (int i = 1; i < _layers.Length; i++)
            {
                Layer l = _layers[i - 1];
                Layer l1 = _layers[i];

                Task[] TaskList = new Task[l1.Size];

                for (int j = 0; j < l1.Size; j++)
                {
                    int jj = j;
                    Task t = new Task(() => CountLayer(l, l1, jj));
                    TaskList[j] = t;
                    t.Start();
                }

                Task.WaitAll(TaskList);
            }
            return _layers[_layers.Length - 1].Neurons;
        }

        private void CountLayer(Layer l, Layer l1, int j)
        {
            l1.Neurons[j] = 0;

            for (int k = 0; k < l.Size; k++)
            {
                l1.Neurons[j] += l.Neurons[k] * l.Weights[k, j];
            }

            l1.Neurons[j] += l1.Biases[j];
            l1.Neurons[j] = _activation(l1.Neurons[j]);
        }

        public void Backpropagation(double[] targets)
        {
            double[] errors = new double[_layers[_layers.Length - 1].Size];

            for (int i = 0; i < _layers[_layers.Length - 1].Size; i++)
            {
                errors[i] = targets[i] - _layers[_layers.Length - 1].Neurons[i];
            }

            for (int k = _layers.Length - 2; k >= 0; k--)
            {
                Layer l = _layers[k];
                Layer l1 = _layers[k + 1];

                double[] errorNext = new double[l.Size];
                double[] gradients = new double[l1.Size];

                for (int i = 0; i < l1.Size; i++)
                {
                    gradients[i] = errors[i] * _derivative(_layers[k + 1].Neurons[i]);
                    gradients[i] *= _learningRate;
                }

                double[,] deltas = new double[l1.Size, l.Size];
                for (int i = 0; i < l1.Size; i++)
                {
                    for (int j = 0; j < l.Size; j++)
                    {
                        deltas[i, j] = gradients[i] * l.Neurons[j];
                    }
                }

                for (int i = 0; i < l.Size; i++)
                {
                    errorNext[i] = 0;
                    for (int j = 0; j < l1.Size; j++)
                    {
                        errorNext[i] += l.Weights[i, j] * errors[j];
                    }
                }

                errors = new double[l.Size];

                Array.Copy(errorNext, 0, errors, 0, l.Size);
                //l.Weights.Length;
                double[,] weightsNew = new double[l.Weights.GetLength(0), l.Weights.GetLength(1)];

                for (int i = 0; i < l1.Size; i++)
                {
                    for (int j = 0; j < l.Size; j++)
                    {
                        weightsNew[j, i] = l.Weights[j, i] + deltas[i, j];
                    }
                }

                l.Weights = weightsNew;
                for (int i = 0; i < l1.Size; i++)
                {
                    l1.Biases[i] += gradients[i];
                }
            }
        }
    }
}
