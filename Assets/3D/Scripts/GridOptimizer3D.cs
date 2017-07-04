using UnityEngine;
using System.Runtime.InteropServices;

namespace Kodai.GridNeighborSearch3D {
    public class GridOptimizer3D<T> : GridOptimizerBase where T : struct {
        
        private Vector3 gridDim;

        #region Accessor
        public float GetGridH() {
            return gridH;
        }

        public ComputeBuffer GetGridIndicesBuffer() {
            return gridIndicesBuffer;
        }
        #endregion

        public GridOptimizer3D(int numObjects, Vector3 range, Vector3 dimension, ComputeShader bitonic, ComputeShader gridSort) : base(numObjects, bitonic, gridSort) {
            this.gridDim = dimension;
            this.numGrid = (int)(dimension.x * dimension.y * dimension.z);
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