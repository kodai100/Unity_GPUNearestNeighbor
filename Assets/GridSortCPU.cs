using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CPU {

    public class GridSortCPU : MonoBehaviour {

        int _NumParticles;
        Vector2 _Range;
        Vector2 _GridDim;
        float _GridH;

        void Start() {

        }

        void Update() {

        }

        Vector2 GridCalculateCell(Vector2 pos) {
            return pos / _GridH;
        }

        uint GridKey(Vector2 xy) {
            return (uint)(xy.x + xy.y * _GridDim.x);
        }

        //Uint2 MakeKeyValuePair(Vector2 xy, uint value) {
        //    return Uint2(GridKey(xy), value); ;
        //}

    }

    public struct ParticleCPU {
        Vector2 pos;
        Color col;
    }

    public struct Uint2 {
        public uint x;
        public uint y;
        public Uint2(uint x, uint y) {
            this.x = x;
            this.y = y;
        }
    }

}