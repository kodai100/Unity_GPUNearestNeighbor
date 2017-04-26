using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Kodai.NeighborSearch {

    public class GridNeighborSearchCS : MonoBehaviour {

        public ComputeShader BitonicCS;
        public ComputeShader GridSortCS;

        ComputeBuffer particlesBufferRead;          // Particles Buffer (Position)
        ComputeBuffer gridBuffer;               // Pair Of G_ID and P_ID
        ComputeBuffer gridPingPongBuffer;
        ComputeBuffer gridIndicesBuffer;        // Indices of Grid ID Starts and ends
        ComputeBuffer sortedParticlesBuffer;    // Grid Sorted Particles Data
        ComputeBuffer particlesBufferWrite;

        private static int SIMULATION_BLOCK_SIZE = 32;
        private int threadGroupSize;

        static uint BITONIC_BLOCK_SIZE = 512;
        static uint TRANSPOSE_BLOCK_SIZE = 16;

        public int maxParticleNum = 256;

        public Vector2 range = new Vector2(128,128);
        public Vector2 gridDim = new Vector2(16, 16);

        private int numGrid;
        private int gridH;
        Particle[] particles;

        void Start() {
            numGrid = (int)(gridDim.x * gridDim.y);
            gridH = (int)(range.x / gridDim.x);
            InitializeBuffer();
            InitializeParticle();
        }
        
        void DebugGrid(int flag) {

            Uint2[] a = new Uint2[maxParticleNum];
            gridBuffer.GetData(a);
            string str = "";
            string str2 = "";
            for (int i = 0; i < maxParticleNum; i++) {
                str += a[i].x + ",";
                str2 += a[i].y + ",";
            }

            // Before
            if(flag == 0) {
                Debug.Log("<color='cyan'>Grid ID: " + str + "</color>");
                Debug.Log("<color='cyan'>Particle ID: " + str2 + "</color>");
            } else {
                Debug.Log("<color='red'>Grid ID: " + str + "</color>");
                Debug.Log("<color='red'>Particle ID: " + str2 + "</color>");
            }
            
        }

        void DebugGridIndices() {
            Uint2[] b = new Uint2[gridIndicesBuffer.count];
            gridIndicesBuffer.GetData(b);
            string str3 = "";
            string str4 = "";
            for (int i = 0; i < gridIndicesBuffer.count; i++) {
                str3 += b[i].x.ToString("000") + ",";
                str4 += b[i].y.ToString("000") + ",";
            }
            Debug.Log("s: " + str3);
            Debug.Log("e: " + str4);
        }

        void Update() {
            // Set Variables
            GridSortCS.SetInt("_NumParticles", maxParticleNum);
            GridSortCS.SetVector("_Range", range);
            GridSortCS.SetVector("_GridDim", gridDim);
            GridSortCS.SetFloat("_GridH", gridH);
            
            int kernel = 0;

            // -----------------------------------------------------------------
            // Build Grid : 粒子の位置からグリッドハッシュとパーティクルIDを結びつける
            // -----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("BuildGridCS");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_GridBufferWrite", gridBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            //DebugGrid(0);

            // -----------------------------------------------------------------
            // Sort Grid : グリッドインデックス順に粒子インデックスをソートする
            // -----------------------------------------------------------------
            // GPUSort(gridBuffer, gridPingPongBuffer);

            CPUSort();

            //DebugGrid(1);

            // -----------------------------------------------------------------
            // Build Grid Indices : グリッドの開始終了インデックスを格納
            // -----------------------------------------------------------------
            // 初期化
            kernel = GridSortCS.FindKernel("ClearGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, Mathf.CeilToInt(numGrid / SIMULATION_BLOCK_SIZE)+1, 1, 1);

            // 格納
            kernel = GridSortCS.FindKernel("BuildGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            //DebugGridIndices();

            //　-----------------------------------------------------------------
            // Rearrange : ソートしたグリッド関連付け配列からパーティクルIDだけを取り出す
            //　-----------------------------------------------------------------
            kernel = GridSortCS.FindKernel("RearrangeParticlesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", sortedParticlesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            // Update
            kernel = GridSortCS.FindKernel("Update");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", sortedParticlesBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferWrite);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferRead", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer(ref particlesBufferRead, ref particlesBufferWrite);
        }

        void OnDrawGizmos() {
            Gizmos.DrawWireCube(range / 2, range);

            Gizmos.color = Color.blue;
            for (int i = 1; i < gridDim.y; i++) {
                Gizmos.DrawLine(new Vector3(0, gridH * i, 0), new Vector3(range.x, gridH * i, 0));
            }

            for (int i = 1; i < gridDim.x; i++) {
                Gizmos.DrawLine(new Vector3(gridH * i, 0, 0), new Vector3(gridH * i, range.y, 0));
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
            for(int i = 0; i < maxParticleNum; i++) {
                particles[i] = new Particle(new Vector2(Random.Range(1, range.x), Random.Range(1, range.y)));
            }
            threadGroupSize = Mathf.CeilToInt(maxParticleNum / SIMULATION_BLOCK_SIZE) + 1;
            particlesBufferRead.SetData(particles);
            particlesBufferWrite.SetData(particles);
        }

        public ComputeBuffer GetBuffer() {
            return particlesBufferRead;
        }

        public int GetMaxParticleNum() {
            return maxParticleNum;
        }

        void CPUSort() {
            Uint2[] a = new Uint2[maxParticleNum];
            gridBuffer.GetData(a);
            for (int i = 0; i < maxParticleNum - 1; i++) {
                // 下から上に順番に比較します
                for (int j = maxParticleNum - 1; j > i; j--) {
                    // 上の方が大きいときは互いに入れ替えます
                    if (a[j].x < a[j - 1].x) {
                        Uint2 t = a[j];
                        a[j] = a[j - 1];
                        a[j - 1] = t;
                    }
                }
            }
            gridBuffer.SetData(a);
        }

        void GPUSort(ComputeBuffer inBuffer, ComputeBuffer tempBuffer) {
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

public struct Particle {
    public Vector2 oldPos;
    public Vector2 newPos;
    public Vector3 color;

    public Particle(Vector2 pos) {
        this.oldPos = pos;
        this.newPos = pos;
        this.color = new Vector3(1, 1, 1);
    }
}

public struct Uint2 {
    public uint x;
    public uint y;
}
