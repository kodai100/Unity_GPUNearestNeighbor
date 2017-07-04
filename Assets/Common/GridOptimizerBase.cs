using UnityEngine;

public abstract class GridOptimizerBase {

    protected ComputeBuffer gridBuffer;
    protected ComputeBuffer gridPingPongBuffer;
    protected ComputeBuffer gridIndicesBuffer;
    protected ComputeBuffer sortedObjectsBufferOutput;

    protected int numObjects;

    protected ComputeShader BitonicCS;    // I want to assign this in inspector
    protected ComputeShader GridSortCS;   // I want to assign this in inspector
    protected static readonly int SIMULATION_BLOCK_SIZE_FOR_GRID = 32;
    protected static readonly uint BITONIC_BLOCK_SIZE = 512;
    protected static readonly uint TRANSPOSE_BLOCK_SIZE = 16;

    protected int threadGroupSize;
    protected int numGrid;
    protected float gridH;

    public GridOptimizerBase(int numObjects, ComputeShader bitonic, ComputeShader gridSort) {
        this.numObjects = numObjects;
        this.BitonicCS = bitonic;
        this.GridSortCS = gridSort;
        this.threadGroupSize = numObjects / SIMULATION_BLOCK_SIZE_FOR_GRID;
    }

    public void Release() {
        DestroyBuffer(gridBuffer);
        DestroyBuffer(gridIndicesBuffer);
        DestroyBuffer(gridPingPongBuffer);
        DestroyBuffer(sortedObjectsBufferOutput);
    }

    void DestroyBuffer(ComputeBuffer buffer) {
        if (buffer != null) {
            buffer.Release();
            buffer = null;
        }
    }

    public void GridSort(ref ComputeBuffer objectsBufferInput) {

        GridSortCS.SetInt("_NumParticles", numObjects);
        SetCSVariables();

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

    protected abstract void InitializeBuffer();

    protected abstract void SetCSVariables();
}
