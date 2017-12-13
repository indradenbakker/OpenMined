﻿using System;
using System.Linq;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.NN
{
    public static class Functional
    {
        public static FloatTensor Softmax(FloatTensor input, long dim = -1)
        {
            
            // TODO -- GPU Support
            
            var copy = input.emptyTensorCopy();
            if (dim == -1)
            {
                dim = input.strides.Length - 1;
            }

            input.ForEach(dim, (vals, offset, stride) =>
            {
                var sum = vals.Sum(d => (float) Math.Pow(Math.E, d));
                for (var v = 0; v < vals.Length; ++v)
                {
                    copy.data[offset + v * stride] = (float) Math.Pow(Math.E, input.data[offset + v * stride]) / sum;
                }
            });
			
            return copy;
        }
    }
}