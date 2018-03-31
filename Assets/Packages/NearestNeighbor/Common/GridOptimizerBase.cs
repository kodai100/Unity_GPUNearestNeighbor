using UnityEngine;
using Sorting.BitonicSort;

public abstract class GridOptimizerBase {

    protected ComputeBuffer gridBuffer;
    protected ComputeBuffer gridPingPongBuffer;
    protected ComputeBuffer gridIndicesBuffer;
    protected ComputeBuffer sortedObjectsBufferOutput;

    protected int numObjects;

    BitonicSort bitonicSort;

   
    protected ComputeShader GridSortCS;
    protected static readonly int SIMULATION_BLOCK_SIZE_FOR_GRID = 32;
    

    protected int threadGroupSize;
    protected int numGrid;
    protected float gridH;

    public GridOptimizerBase(int numObjects) {
        this.numObjects = numObjects;

        this.threadGroupSize = numObjects / SIMULATION_BLOCK_SIZE_FOR_GRID;

        bitonicSort = new BitonicSort(numObjects);
    }

    #region Accessor
    public float GetGridH() {
        return gridH;
    }

    public ComputeBuffer GetGridIndicesBuffer() {
        return gridIndicesBuffer;
    }
    #endregion

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
        bitonicSort.Sort(ref gridBuffer, ref gridPingPongBuffer);

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
    
    #endregion GPUSort 

    protected abstract void InitializeBuffer();

    protected abstract void SetCSVariables();
}
