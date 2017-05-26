using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Kodai.GridNeighborSearch3D {

    public class GridNeighborSearchCS : MonoBehaviour {

        public ComputeShader BitonicCS;
        public ComputeShader GridSortCS;
        public bool debugConsole; 

        ComputeBuffer particlesBufferRead;
        ComputeBuffer gridBuffer;
        ComputeBuffer gridPingPongBuffer;
        ComputeBuffer gridIndicesBuffer;
        ComputeBuffer sortedParticlesBuffer;
        ComputeBuffer particlesBufferWrite;

        private static int SIMULATION_BLOCK_SIZE = 32;
        private int threadGroupSize;

        static uint BITONIC_BLOCK_SIZE = 512;
        static uint TRANSPOSE_BLOCK_SIZE = 16;

        private int maxParticleNum;
        public enum Mode {
            NUM_8K, NUM_16K, NUM_32K, NUM_65K, NUM_130K, NUM_260K
        }
        public Mode num;

        public int dispIdx;

        public Vector3 range = new Vector3(128, 128, 128);
        public Vector3 gridDim = new Vector3(16, 16, 16);

        private int numGrid;
        private int gridH;
        Particle[] particles;

        void Start() {
            numGrid = (int)(gridDim.x * gridDim.y * gridDim.z);
            gridH = (int)(range.x / gridDim.x);

            switch (num) {
                case Mode.NUM_8K:
                    maxParticleNum = 8192;
                    break;
                case Mode.NUM_16K:
                    maxParticleNum = 16384;
                    break;
                case Mode.NUM_32K:
                    maxParticleNum = 32768;
                    break;
                case Mode.NUM_65K:
                    maxParticleNum = 65536;
                    break;
                case Mode.NUM_130K:
                    maxParticleNum = 131072;
                    break;
                case Mode.NUM_260K:
                    maxParticleNum = 262144;
                    break;
                default:
                    maxParticleNum = 8192;
                    break;
            }

            InitializeBuffer();
            InitializeParticle();

            Debug.Log("Initiated.");
        }

        void Update() {
            GridSortCS.SetInt("_NumParticles", maxParticleNum);
            GridSortCS.SetVector("_Range", range);
            GridSortCS.SetVector("_GridDim", gridDim);
            GridSortCS.SetFloat("_GridH", gridH);
            GridSortCS.SetInt("_DispIdx", dispIdx);

            int kernel = 0;
            
            kernel = GridSortCS.FindKernel("BuildGridCS");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_GridBufferWrite", gridBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);
            
            GPUSort(ref gridBuffer, ref gridPingPongBuffer);
            DebugConsole("Sorted", maxParticleNum, gridBuffer);

            kernel = GridSortCS.FindKernel("ClearGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, (int)(numGrid / SIMULATION_BLOCK_SIZE), 1, 1);

            kernel = GridSortCS.FindKernel("BuildGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);
            DebugConsole("Indices", numGrid, gridIndicesBuffer);
            
            kernel = GridSortCS.FindKernel("RearrangeParticlesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", sortedParticlesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            kernel = GridSortCS.FindKernel("CopyBuffer");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", sortedParticlesBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferRead);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            kernel = GridSortCS.FindKernel("Update");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferWrite);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferRead", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer(ref particlesBufferRead, ref particlesBufferWrite);
        }

        void DebugConsole(string text, int num, ComputeBuffer cb) {
            if (debugConsole) {
                Uint2[] res = new Uint2[num];
                cb.GetData(res);
                string r1 = "", r2 = "";
                foreach (Uint2 tmp in res) {
                    r1 += tmp.x + ",";
                    r2 += tmp.y + ",";
                }
                Debug.Log(text);
                Debug.Log("<color=yellow>" + r1 + "</color>");
                Debug.Log("<color=cyan>" + r2 + "</color>");
            }
            
        }

        void OnDrawGizmos() {
            Gizmos.DrawWireCube(range / 2, range);

            Gizmos.color = Color.grey;
            for (int z = 0; z < gridDim.z; z++) {
                for (int i = 1; i < gridDim.y; i++) {
                    Gizmos.DrawLine(new Vector3(0, gridH * i, gridH * z), new Vector3(range.x, gridH * i, gridH*z));
                }

                for (int i = 1; i < gridDim.x; i++) {
                    Gizmos.DrawLine(new Vector3(gridH * i, 0, gridH * z), new Vector3(gridH * i, range.y, gridH * z));
                }
            }
        }

        void OnDestroy() {
            if (particlesBufferRead != null) {
                particlesBufferRead.Release();
                particlesBufferRead = null;
            }
            if (particlesBufferWrite != null) {
                particlesBufferWrite.Release();
                particlesBufferWrite = null;
            }
            if (gridBuffer != null) {
                gridBuffer.Release();
                gridBuffer = null;
            }
            if (gridIndicesBuffer != null) {
                gridIndicesBuffer.Release();
                gridIndicesBuffer = null;
            }
            if (gridPingPongBuffer != null) {
                gridPingPongBuffer.Release();
                gridPingPongBuffer = null;
            }
            if (sortedParticlesBuffer != null) {
                sortedParticlesBuffer.Release();
                sortedParticlesBuffer = null;
            }
        }

        void InitializeBuffer() {
            particlesBufferRead = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
            gridBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Uint2)));
            gridPingPongBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Uint2)));
            gridIndicesBuffer = new ComputeBuffer(numGrid, Marshal.SizeOf(typeof(Uint2)));
            sortedParticlesBuffer = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
            particlesBufferWrite = new ComputeBuffer(maxParticleNum, Marshal.SizeOf(typeof(Particle)));
        }

        void InitializeParticle() {
            particles = new Particle[maxParticleNum];
            for (int i = 0; i < maxParticleNum; i++) {
                particles[i] = new Particle(new Vector3(Random.Range(1, range.x), Random.Range(1, range.y), Random.Range(1, range.z)));
            }
            threadGroupSize = (int)(maxParticleNum / SIMULATION_BLOCK_SIZE);
            particlesBufferRead.SetData(particles);
            particlesBufferWrite.SetData(particles);
        }

        public ComputeBuffer GetBuffer() {
            return particlesBufferRead;
        }

        public int GetMaxParticleNum() {
            return maxParticleNum;
        }

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

        void SwapBuffer(ref ComputeBuffer src, ref ComputeBuffer dst) {
            ComputeBuffer tmp = src;
            src = dst;
            dst = tmp;
        }
    }


}
