using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Kodai.NeighborSearch {

    [System.Serializable]
    public class NeighborSearchSetting<Dim, Vec> where Dim : IGridVector<Vec> where Vec : struct{

        private GridOptimizer<MyParticleData<Vec>, Dim, Vec> gridOptimizer;
        public GridOptimizer<MyParticleData<Vec>, Dim, Vec> GridOptimizer {
            get { return gridOptimizer; }
        }

        [SerializeField] public Vec particleScatterRange;
        [SerializeField] public Dim gridDimension;

        public NeighborSearchSetting() { }

        public NeighborSearchSetting(ParticleNumEnum num, Vec range, Dim dim) {
            particleScatterRange = range;
            gridDimension = dim;
            gridOptimizer = new GridOptimizer<MyParticleData<Vec>, Dim, Vec>(num, particleScatterRange, gridDimension);
        }
    }

    [System.Serializable]
    public class NeighborSearchSetting2D : NeighborSearchSetting<IntDim2D, Vector2> {

        public NeighborSearchSetting2D() { }

        public NeighborSearchSetting2D(ParticleNumEnum num, Vector2 range, IntDim2D dim) : base(num, range, dim) {}
    }

    [System.Serializable]
    public class NeighborSearchSetting3D : NeighborSearchSetting<IntDim3D, Vector3> {

        public NeighborSearchSetting3D() { }

        public NeighborSearchSetting3D(ParticleNumEnum num, Vector3 range, IntDim3D dim) : base(num, range, dim) { }
    }

}