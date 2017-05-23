using UnityEngine;

namespace Kodai.GridNeighborSearch3D {

    public struct Particle {
        public Vector3 oldPos;
        public Vector3 newPos;
        public Vector3 color;

        public Particle(Vector3 pos) {
            this.oldPos = pos;
            this.newPos = pos;
            this.color = new Vector3(1, 1, 1);
        }
    }
}