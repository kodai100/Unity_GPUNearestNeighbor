using UnityEngine;

namespace Kodai.GridNeighborSearch3D {

    public struct Particle {
        public Vector3 pos;
        public Vector3 color;

        public Particle(Vector3 pos) {
            this.pos = pos;
            this.color = new Vector3(1, 1, 1);
        }
    }
}