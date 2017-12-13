using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenMined.Network.Controllers;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public abstract partial class BaseTensor<T>
    {
        protected static volatile int nCreated = 0;

        protected SyftController ctrl;

        protected T[] data;
        protected int id;
        protected int[] strides;
        protected int[] shape;
        protected int size;


        public T[] Data => data;

        public int[] Shape => shape;

        public int[] Strides => strides;

        public int Size => size;

        public int Id => id;

        public static int CreatedObjectCount => nCreated;

        public void InitCpu(T[] _data, bool _copyData)
        {
            // looks like we have CPU data being passed in... initialize a CPU tensor.
            dataOnGpu = false;

            if (_copyData)
            {
                data = (T[]) _data.Clone();
            }
            else
            {
                data = _data;
            }
        }

        public void InitGpu(ComputeShader _shader, ComputeBuffer _dataBuffer, ComputeBuffer _shapeBuffer,
            bool _copyData)
        {
            if (!SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("Shaders are not supported on the host machine");

            if (_copyData)
            {
                var tempData = new T[_dataBuffer.count];
                var tempShape = new int[shape.Length];

                _dataBuffer.GetData(tempData);
                _shapeBuffer.GetData(tempShape);
                dataBuffer = new ComputeBuffer(_dataBuffer.count, Marshal.SizeOf(default(T)));
                shapeBuffer = new ComputeBuffer(_shapeBuffer.count, sizeof(int));

                dataBuffer.SetData(tempData);
                shapeBuffer.SetData(tempShape);
            }

            // Third: let's set the tensor's size to be equal to that of the buffer
            size = _dataBuffer.count;
        }

        protected int GetIndex(params int[] indices)
        {
            var offset = 0;
            for (var i = 0; i < indices.Length; ++i)
            {
                if (indices[i] >= shape[i] || indices[i] < 0)
                    throw new IndexOutOfRangeException();
                offset += indices[i] * strides[i];
            }
            return offset;
        }

        protected long[] GetIndices(long index)
        {
            var idx = index;
            var indices = new long[Shape.Length];
            for (var i = 0; i < Shape.Length; ++i)
            {
                indices[i] = (idx - (idx % (strides[i]))) / strides[i];
                idx -= indices[i] * strides[i];
            }
            return indices;
        }

        public T this[params int[] indices]
        {
            get { return this[GetIndex(indices)]; }
            set { this[GetIndex(indices)] = value; }
        }

        public T this[int index]
        {
            get { return Data[index]; }
            set { Data[index] = value; }
        }
    }
}