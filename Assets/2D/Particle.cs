using UnityEngine;

namespace Kodai.GridNeighborSearch2D {

    public struct Particle {
        public Vector2 pos;
        public Vector3 color;

        public Particle(Vector2 pos) {
            this.pos = pos;
            this.color = new Vector3(1, 1, 1);
        }
    }
}