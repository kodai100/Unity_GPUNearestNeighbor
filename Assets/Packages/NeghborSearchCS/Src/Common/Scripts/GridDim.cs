using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Kodai.NeighborSearch {

    public interface IGridVector<T> {
        GridType GetGridType();
        float GetGridH(T range);
        float OwnMultiply();
        Vector3 ToVector3();
    }

    [System.Serializable]
    public struct IntDim3D : IGridVector<Vector3> {

        public static GridType GridType = GridType.Grid3D;

        public int x;
        public int y;
        public int z;

        public IntDim3D(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public IntDim3D(int x) : this(x, x, x) { }

        public float OwnMultiply() {
            return x * y * z;
        }

        public GridType GetGridType() {
            return GridType;
        }

        public float GetGridH(Vector3 range) {
            return (float)range.x / x;
        }

        public Vector3 ToVector3() {
            return new Vector3(x, y, z);
        }
    }

    [System.Serializable]
    public struct IntDim2D : IGridVector<Vector2> {

        public static GridType GridType = GridType.Grid2D;

        public int x;
        public int y;

        public IntDim2D(int x, int y) {
            this.x = x;
            this.y = y;
        }

        public IntDim2D(int x) : this(x, x) { }

        public float OwnMultiply() {
            return x * y;
        }

        public GridType GetGridType() {
            return GridType;
        }

        public float GetGridH(Vector2 range) {
            return (float)range.x / x;
        }

        public Vector3 ToVector3() {
            return new Vector3(x, y, 0);
        }
    }

}