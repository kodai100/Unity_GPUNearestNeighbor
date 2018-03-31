using System.Runtime.InteropServices;
using UnityEngine;
using NearestNeighbor;

namespace NearestNeighborSample.ThreeDimension
{
    /// <summary>
    /// Define your data 
    /// </summary>
    public struct MyParticle
    {
        public Vector3 pos;
        public Vector3 color;
        public MyParticle(Vector3 pos)
        {
            this.pos = pos;
            this.color = new Vector3(1, 1, 1);
        }
    }

    /// <summary>
    /// Particle system with Grid Optimization
    /// </summary>
    public class MyParticleSystem : MonoBehaviour
    {

        #region ForParticle
        public ComputeShader ParticleCS;

        public PARTICLE_NUM mode = PARTICLE_NUM.NUM_8K;
        public int dispIdx;
        public Material ParticleRenderMat;

        private int threadGroupSize;
        private ComputeBuffer particlesBufferRead;
        private ComputeBuffer particlesBufferWrite;
        private static readonly int SIMULATION_BLOCK_SIZE = 32;
        private int numParticles;
        #endregion ForParticle

        #region ForGrid
        public Vector3 range = new Vector3(128, 128, 128);
        public Vector3 gridDim = new Vector3(16, 16, 16);
        GridOptimizer3D<MyParticle> gridOptimizer;
        #endregion ForGrid

        #region Accessor
        public ComputeBuffer GetBuffer()
        {
            return particlesBufferRead;
        }

        public int GetParticleNum()
        {
            return numParticles;
        }
        #endregion Accessor

        #region MonoBehaviourFuncs
        void Start()
        {
            InitializeVariables();
            InitializeBuffer();
            InitializeParticle();
            InitializeOptimizer();
        }

        void Update()
        {

            // ---- Grid Optimization -------------------------------------------------------------------
            gridOptimizer.GridSort(ref particlesBufferRead);    // Pass the buffer you want to optimize  
                                                                // ---- Grid Optimization -------------------------------------------------------------------


            // ---- Your Particle Process -------------------------------------------------------------------
            ParticleCS.SetInt("_NumParticles", numParticles);
            ParticleCS.SetVector("_GridDim", gridDim);
            ParticleCS.SetInt("_DispIdx", (int)(dispIdx * numParticles * 0.001f));
            ParticleCS.SetFloat("_GridH", gridOptimizer.GetGridH());

            int kernel = ParticleCS.FindKernel("Update");
            ParticleCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            ParticleCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferWrite);
            ParticleCS.SetBuffer(kernel, "_GridIndicesBufferRead", gridOptimizer.GetGridIndicesBuffer());   // Get and use a GridIndicesBuffer to find neighbor
            ParticleCS.Dispatch(kernel, threadGroupSize, 1, 1);
            // ---- Your Particle Process -------------------------------------------------------------------

            SwapBuffer(ref particlesBufferRead, ref particlesBufferWrite);
        }

        private void OnRenderObject()
        {
            Material m = ParticleRenderMat;
            m.SetPass(0);
            m.SetBuffer("_Particles", GetBuffer());
            Graphics.DrawProcedural(MeshTopology.Points, GetParticleNum());
        }

        void OnDestroy()
        {
            DestroyBuffer(particlesBufferRead);
            DestroyBuffer(particlesBufferWrite);
            gridOptimizer.Release();                // Must
        }
        #endregion MonoBehaviourFuncs

        #region PrivateFuncs
        void InitializeVariables()
        {
            numParticles = (int)mode;
        }

        void InitializeBuffer()
        {
            particlesBufferRead = new ComputeBuffer(numParticles, Marshal.SizeOf(typeof(MyParticle)));
            particlesBufferWrite = new ComputeBuffer(numParticles, Marshal.SizeOf(typeof(MyParticle)));
        }

        void InitializeParticle()
        {
            MyParticle[] particles = new MyParticle[numParticles];
            for (int i = 0; i < numParticles; i++)
            {
                particles[i] = new MyParticle(new Vector3(Random.Range(1, range.x), Random.Range(1, range.y), Random.Range(1, range.z)));
            }
            threadGroupSize = numParticles / SIMULATION_BLOCK_SIZE;
            particlesBufferRead.SetData(particles);
        }

        void InitializeOptimizer()
        {
            gridOptimizer = new GridOptimizer3D<MyParticle>(numParticles, range, gridDim);
        }

        void SwapBuffer(ref ComputeBuffer src, ref ComputeBuffer dst)
        {
            ComputeBuffer tmp = src;
            src = dst;
            dst = tmp;
        }

        void DestroyBuffer(ComputeBuffer buffer)
        {
            if (buffer != null)
            {
                buffer.Release();
                buffer = null;
            }
        }

        #endregion PrivateFuncs
    }


}