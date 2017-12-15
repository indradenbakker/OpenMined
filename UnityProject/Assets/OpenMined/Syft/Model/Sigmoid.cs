﻿using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Sigmoid: Model
    {
		
        public Sigmoid (SyftController controller)
        {
            init();
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }
        
        public override FloatTensor Forward(FloatTensor input)
        {
            return input.Sigmoid();
        }
    }
}