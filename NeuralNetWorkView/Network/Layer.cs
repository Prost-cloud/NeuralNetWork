using System;

namespace NeuralNetwork.Network
{
    public class Layer
    {
        public int Size { get; private set; }
        public double[] Neurons { get; set; }
        public double[] Biases { get; set; }
        public double[,] Weights { get; set; }

        public Layer(int size, int nextSize)
        {
            Size = size;    
            Neurons = new double[size];
            Biases = new double[size];
            Weights = new double[size, nextSize];
        }

    }
}
