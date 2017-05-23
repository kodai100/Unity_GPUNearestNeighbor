using System.Runtime.InteropServices;
using UnityEngine;

namespace Kodai.GridNeighborSearch2D {

    public class GridNeighborSearchCS : MonoBehaviour {

        #region ForGrid
        public ComputeShader BitonicCS;
        public ComputeShader GridSortCS;
        public Vector2 range = new Vector2(128, 128);
        public Vector2 gridDim = new Vector2(16, 16);
        ComputeBuffer particlesBufferRead;
        ComputeBuffer gridBuffer;
        ComputeBuffer gridPingPongBuffer;
        ComputeBuffer gridIndicesBuffer;
        ComputeBuffer sortedParticlesBuffer;
        ComputeBuffer particlesBufferWrite;
        private static int SIMULATION_BLOCK_SIZE_FOR_GRID = 32;
        static uint BITONIC_BLOCK_SIZE = 512;
        static uint TRANSPOSE_BLOCK_SIZE = 16;
        private int threadGroupSize;
        private int numGrid;
        private int gridH;
        #endregion ForGrid

        #region ForParticle
        public ComputeShader ParticleCS;
        public enum Mode {
            NUM_8K = 8192, NUM_16K = 16384, NUM_32K = 32768, NUM_65K = 65536, NUM_130K = 131072, NUM_260K = 262144 
        }
        public Mode mode = Mode.NUM_8K;
        public int dispIdx;
        private static int SIMULATION_BLOCK_SIZE = 32;
        private int maxParticleNum;
        private Particle[] particles;
        #endregion ForParticle

        #region ForThisClass
        public bool drawGrid = true;
        public Color gridColor = Color.blue;
        #endregion ForThisClass

        #region Accessor
        public ComputeBuffer GetBuffer() {
            return particlesBufferRead;
        }

        public int GetMaxParticleNum() {
            return maxParticleNum;
        }
        #endregion Accessor

        #region MonoBehaviourFuncs
        void Start() {
            InitializeVariables();  // Recommended
            InitializeBuffer();
            InitializeParticle();
        }
        
        void Update() {

            GridSortCS.SetInt("_NumParticles", maxParticleNum);
            GridSortCS.SetVector("_GridDim", gridDim);
            GridSortCS.SetFloat("_GridH", gridH);

            int kernel = 0;

            #region GridOptimization
            //-----------------------------------------------------------------
            // Build Grid
            //-----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("BuildGridCS");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_GridBufferWrite", gridBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            //-----------------------------------------------------------------
            // Sort Grid
            //-----------------------------------------------------------------
            GPUSort(ref gridBuffer, ref gridPingPongBuffer);

            //-----------------------------------------------------------------
            // Build Grid Indices
            //-----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("ClearGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, (int)(numGrid / SIMULATION_BLOCK_SIZE_FOR_GRID), 1, 1);
            
            kernel = GridSortCS.FindKernel("BuildGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            //-----------------------------------------------------------------
            // Rearrange
            //-----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("RearrangeParticlesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", sortedParticlesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            //-----------------------------------------------------------------
            // Copy
            //-----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("CopyBuffer");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", sortedParticlesBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferRead);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            #endregion GridOptimization

            //-----------------------------------------------------------------
            // !!!!!!!!!!!!! Update !!!!!!!!!!!!!!! : Write own processes
            //-----------------------------------------------------------------
            ParticleCS.SetInt("_NumParticles", maxParticleNum);
            ParticleCS.SetVector("_GridDim", gridDim);
            ParticleCS.SetInt("_DispIdx", dispIdx);
            ParticleCS.SetFloat("_GridH", gridH);

            kernel = ParticleCS.FindKernel("Update");
            ParticleCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            ParticleCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferWrite);
            ParticleCS.SetBuffer(kernel, "_GridIndicesBufferRead", gridIndicesBuffer);
            ParticleCS.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer(ref particlesBufferRead, ref particlesBufferWrite);
        }

        void OnDrawGizmos() {
            if (drawGrid) {
                Gizmos.DrawWireCube(range / 2, range);

                Gizmos.color = gridColor;
                for (int i = 1; i < gridDim.y; i++) {
                    Gizmos.DrawLine(new Vector3(0, gridH * i, 0), new Vector3(range.x, gridH * i, 0));
                }

                for (int i = 1; i < gridDim.x; i++) {
                    Gizmos.DrawLine(new Vector3(gridH * i, 0, 0), new Vector3(gridH * i, range.y, 0));
                }
            }
        }

        void OnDestroy() {
            DestroyBuffer(particlesBufferRead);
            DestroyBuffer(particlesBufferWrite);
            DestroyBuffer(gridBuffer);
            DestroyBuffer(gridIndicesBuffer);
            DestroyBuffer(gridPingPongBuffer);
            DestroyBuffer(sortedParticlesBuffer);
        }
        #endregion MonoBehaviourFuncs

        #region PrivateFuncs
        void InitializeVariables() {
            maxParticleNum = (int)mode;
            numGrid = (int)(gridDim.x * gridDim.y);
            gridH = (int)(range.x / gridDim.x);
        }

        void InitializeBuffer() {
            particlesBufferRead = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
            particlesBufferWrite = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
            gridBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Uint2)));
            gridPingPongBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Uint2)));
            gridIndicesBuffer = new ComputeBuffer(numGrid, Marshal.SizeOf(typeof(Uint2)));
            sortedParticlesBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
        }

        void InitializeParticle() {
            particles = new Particle[maxParticleNum];
            for(int i = 0; i < maxParticleNum; i++) {
                particles[i] = new Particle(new Vector2(Random.Range(1, range.x), Random.Range(1, range.y)));
            }
            threadGroupSize = maxParticleNum / SIMULATION_BLOCK_SIZE;
            particlesBufferRead.SetData(particles);
        }

        void SwapBuffer(ref ComputeBuffer src, ref ComputeBuffer dst) {
            ComputeBuffer tmp = src;
            src = dst;
            dst = tmp;
        }
        void DestroyBuffer(ComputeBuffer buffer) {
            if (buffer != null) {
                buffer.Release();
                buffer = null;
            }
        }

        #region GPUSort
        void GPUSort(ref ComputeBuffer inBuffer, ref ComputeBuffer tempBuffer) {
            ComputeShader sortCS = BitonicCS;

            int KERNEL_ID_BITONICSORT = sortCS.FindKernel("BitonicSort");
            int KERNEL_ID_TRANSPOSE = sortCS.FindKernel("MatrixTranspose");

            uint NUM_ELEMENTS = (uint)maxParticleNum;
            uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
            uint MATRIX_HEIGHT = (uint)NUM_ELEMENTS / BITONIC_BLOCK_SIZE;

            for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level <<= 1) {
                SetGPUSortConstants(sortCS, level, level, MATRIX_HEIGHT, MATRIX_WIDTH);

                // Sort the row data
                sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
                sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
            }

            // Then sort the rows and columns for the levels > than the block size
            // Transpose. Sort the Columns. Transpose. Sort the Rows.
            for (uint level = (BITONIC_BLOCK_SIZE << 1); level <= NUM_ELEMENTS; level <<= 1) {
                // Transpose the data from buffer 1 into buffer 2
                SetGPUSortConstants(sortCS, level / BITONIC_BLOCK_SIZE, (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT);
                sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Input", inBuffer);
                sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Data", tempBuffer);
                sortCS.Dispatch(KERNEL_ID_TRANSPOSE, (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), 1);

                // Sort the transposed column data
                sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", tempBuffer);
                sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);

                // Transpose the data from buffer 2 back into buffer 1
                SetGPUSortConstants(sortCS, BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH);
                sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Input", tempBuffer);
                sortCS.SetBuffer(KERNEL_ID_TRANSPOSE, "Data", inBuffer);
                sortCS.Dispatch(KERNEL_ID_TRANSPOSE, (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), 1);

                // Sort the row data
                sortCS.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
                sortCS.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
            }
        }

        void SetGPUSortConstants(ComputeShader cs, uint level, uint levelMask, uint width, uint height) {
            cs.SetInt("_Level", (int)level);
            cs.SetInt("_LevelMask", (int)levelMask);
            cs.SetInt("_Width", (int)width);
            cs.SetInt("_Height", (int)height);
        }
        #endregion GPUSort 

        #endregion PrivateFuncs
    }


}
