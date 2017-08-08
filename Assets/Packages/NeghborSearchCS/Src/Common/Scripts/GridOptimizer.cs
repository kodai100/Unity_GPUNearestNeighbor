using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Kodai.NeighborSearch {

    /// Bitonic sort can handle 2^X only
    public enum ParticleNumEnum {
        NUM_8K = 8192, NUM_16K = 16384, NUM_32K = 32768, NUM_65K = 65536, NUM_130K = 131072, NUM_260K = 262144
    }

    public enum GridType { Grid2D, Grid3D };

    [System.Serializable]
    public class GridOptimizer<Data, Dim, Vec> : IDisposable where Data : struct where Dim : IGridVector<Vec> where Vec : struct{

        private ComputeBuffer gridBuffer;
        private ComputeBuffer gridPingPongBuffer;
        private ComputeBuffer gridIndicesBuffer;
        private ComputeBuffer sortedObjectsBufferOutput;
        
        private int numObjects;

        private ComputeShader BitonicCS;
        private ComputeShader GridSortCS;
        private static readonly int SIMULATION_BLOCK_SIZE_FOR_GRID = 32;
        private static readonly uint BITONIC_BLOCK_SIZE = 512;
        private static readonly uint TRANSPOSE_BLOCK_SIZE = 16;

        private int threadGroupSize;
        private int numGrid;
        private float gridH;

        private Dim gridDim;

        #region Accessor
        public float GridH {
            get { return gridH; }
        }

        public ComputeBuffer GridIndicesBuffer {
            get { return gridIndicesBuffer; }
        }
        #endregion

        public GridOptimizer(ParticleNumEnum numParticleEnum, Vec range, Dim dimension) {
            numObjects = (int)numParticleEnum;
            threadGroupSize = numObjects / SIMULATION_BLOCK_SIZE_FOR_GRID;

            // 名前からコンピュートシェーダを取得
            BitonicCS = (ComputeShader)Resources.Load("BitonicSort");
            GridSortCS = (ComputeShader)Resources.Load(Enum.GetName(typeof(GridType), dimension.GetGridType()));
           
            gridDim = dimension;
            numGrid = (int) dimension.OwnMultiply();
            gridH = dimension.GetGridH(range);

            gridBuffer = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(Uint2)));
            gridPingPongBuffer = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(Uint2)));
            gridIndicesBuffer = new ComputeBuffer(numGrid, Marshal.SizeOf(typeof(Uint2)));
            sortedObjectsBufferOutput = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(Data)));

            Debug.Log("=== Initialized Grid Sort Package === \nRange : " + range + "\nNumGrid : " + numGrid + "\nGridDim : " + gridDim + "\nGridH : " + gridH);
        }
        
        public void Dispose() {
            Release();
        }

        private void Release() {
            DestroyBuffer(gridBuffer);
            DestroyBuffer(gridIndicesBuffer);
            DestroyBuffer(gridPingPongBuffer);
            DestroyBuffer(sortedObjectsBufferOutput);
        }

        private void DestroyBuffer(ComputeBuffer buffer) {
            if (buffer != null) {
                buffer.Release();
                buffer = null;
            }
        }

        /// <summary>
        /// Grid optimization. Please call this function before run your particle process.
        /// </summary>
        /// <param name="objectsBufferInput">Your particle data. Returns sorted buffer and make indices buffer (you can get from GridIndicesBuffer after this process)</param>
        public void GridSort(ref ComputeBuffer objectsBufferInput) {

            GridSortCS.SetInt("_NumParticles", numObjects);
            GridSortCS.SetVector("_GridDim", gridDim.ToVector3());
            GridSortCS.SetFloat("_GridH", gridH);

            int kernel = 0;

            #region GridOptimization
            // Build Grid
            kernel = GridSortCS.FindKernel("BuildGridCS");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", objectsBufferInput);
            GridSortCS.SetBuffer(kernel, "_GridBufferWrite", gridBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            // Sort Grid
            GPUSort(ref gridBuffer, ref gridPingPongBuffer);

            // Build Grid Indices
            kernel = GridSortCS.FindKernel("ClearGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, (int)(numGrid / SIMULATION_BLOCK_SIZE_FOR_GRID), 1, 1);

            kernel = GridSortCS.FindKernel("BuildGridIndicesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_GridIndicesBufferWrite", gridIndicesBuffer);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

            // Rearrange
            kernel = GridSortCS.FindKernel("RearrangeParticlesCS");
            GridSortCS.SetBuffer(kernel, "_GridBufferRead", gridBuffer);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", objectsBufferInput);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", sortedObjectsBufferOutput);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);
            #endregion GridOptimization

            // Copy buffer
            kernel = GridSortCS.FindKernel("CopyBuffer");
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferRead", sortedObjectsBufferOutput);
            GridSortCS.SetBuffer(kernel, "_ParticlesBufferWrite", objectsBufferInput);
            GridSortCS.Dispatch(kernel, threadGroupSize, 1, 1);

        }

        #region GPUSort
        void GPUSort(ref ComputeBuffer inBuffer, ref ComputeBuffer tempBuffer) {
            ComputeShader sortCS = BitonicCS;

            int KERNEL_ID_BITONICSORT = sortCS.FindKernel("BitonicSort");
            int KERNEL_ID_TRANSPOSE = sortCS.FindKernel("MatrixTranspose");

            uint NUM_ELEMENTS = (uint)numObjects;
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
        
    }
}