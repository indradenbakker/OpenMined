﻿using System;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private bool autograd;
        public FloatTensor Grad { get; private set; }

        private bool keepgrads;

        private List<FloatTensor> creators;
        private string creation_op;
        private Dictionary<int, int> children;

        public void InitAutograd()
        {
//			if(!autograd) {
            autograd = true;
            creators = new List<FloatTensor>();
            children = new Dictionary<int, int>();
//			}
        }

        public bool AllChildrenGradsAccountedFor()
        {
            foreach (var item in children)
            {
                if (item.Value == 0)
                {
                    return false;
                }
            }
            return true;
        }


        // hook autograd two parents - one scalar
        public void HookAutograd(ref FloatTensor result, float x, string creation_op)
        {
            if (autograd)
            {
                FloatTensor new_child =
                    new FloatTensor(_controller: controller, _shape: new int[] {1}, _data: new float[] {x});

                result.InitAutograd();
                result.creators.Add(this);
                result.creators.Add(new_child);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
//				new_child.children.Add (result.Id, 0);
            }
        }

        // hook autograd two parents
        public void HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creators.Add(this);
                result.creators.Add(x);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
                x.children.Add(result.Id, 0);
            }
        }

        // hook autograd single parent
        public void HookAutograd(ref FloatTensor result, string creation_op)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creators.Add(this);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
            }
        }

        public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
        {
            if (!autograd) return;
            if (grad == null)
            {
                grad = this.controller.createOnesTensorLike(this);
                grad.Autograd = false;
            }

            if (grad_origin != null)
            {
                if (children[grad_origin.Id] > 0)
                {
                    throw new InvalidOperationException("Can't backprop more than once.");
                }
                else
                {
                    children[grad_origin.Id] += 1;
                }
            }

            if (this.Grad == null)
            {
                this.Grad = grad;
            }
            else
            {
                this.Grad.Add(grad, true);
            }

            // grads must not have grads of their own
            if (this.Grad.autograd == true)
            {
                throw new InvalidOperationException("Sorry, grads cannot have grads");
            }

            // only continue backpropping if there's something to backprop into
            // only continue backpropping if all gradients (from children) are accounted for
            // override waiting for children if "backprop" was called on this variable directly
            if (this.creators == null || this.creators.Count <= 0 ||
                (grad_origin != null && !AllChildrenGradsAccountedFor())) return;
            if (creation_op == "add_elem")
            {
                creators[0].Backward(grad.Copy(), this);
                creators[1].Backward(grad.Copy(), this);
            }
            else if (creation_op == "mul_elem")
            {
                creators[0].Backward(grad.Mul(creators[1]), this);
                creators[1].Backward(grad.Mul(creators[0]), this);
            }
            else if (creation_op == "div_elem")
            {
                creators[0].Backward(grad.Div(creators[1]), this);
                creators[1].Backward(grad.Div(creators[0]), this);
            }
            else if (creation_op == "sub_elem")
            {
                creators[0].Backward(grad.Copy(), this);
                creators[1].Backward(grad.Neg(), this);
            }
            else if (creation_op == "mm")
            {
                creators[0].Backward(grad.MM(creators[1].Transpose()), this);
                creators[1].Backward(grad.Transpose().MM(creators[0]), this);
            }
            else if (creation_op == "sigmoid")
            {
                var c = this.Copy();
                c.autograd = false;
                creators[0].Backward(c.Neg().Add((float) 1).Mul(this).Mul(grad), this);
            }
            else if (creation_op == "pow_scalar")
            {
                var self_nograd = creators[0].Copy();
                self_nograd.autograd = false;
                creators[0].Backward(self_nograd.Mul(grad).Mul(creators[1][0]), this);
            }

//					if (!keepgrads) {
//						ctrl.RemoveTensor (grad.id);
//					}
        }
    }
}