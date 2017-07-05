using UnityEngine;
using System.Runtime.InteropServices;

namespace Kodai.GridNeighborSearch2D {
    public class GridOptimizer2D<T> : GridOptimizerBase where T : struct {

        private Vector2 gridDim;

        public GridOptimizer2D(int numObjects, Vector2 range, Vector2 dimension, ComputeShader bitonic, ComputeShader gridSort) : base(numObjects, bitonic, gridSort) {
            this.gridDim = dimension;
            this.numGrid = (int)(dimension.x * dimension.y);
            this.gridH = range.x / gridDim.x;

            InitializeBuffer();

            Debug.Log("=== Instantiated Grid Sort === \nRange : " + range + "\nNumGrid : " + numGrid + "\nGridDim : " + gridDim + "\nGridH : " + gridH);
        }

        protected override void InitializeBuffer() {
            gridBuffer = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(Uint2)));
            gridPingPongBuffer = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(Uint2)));
            gridIndicesBuffer = new ComputeBuffer(numGrid, Marshal.SizeOf(typeof(Uint2)));
            sortedObjectsBufferOutput = new ComputeBuffer(numObjects, Marshal.SizeOf(typeof(T)));
        }

        protected override void SetCSVariables() {
            GridSortCS.SetVector("_GridDim", gridDim);
            GridSortCS.SetFloat("_GridH", gridH);
        }

    }
}