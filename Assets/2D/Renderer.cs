using UnityEngine;

namespace Kodai.GridNeighborSearch2D {

    public class Renderer : MonoBehaviour {

        public GridNeighborSearchCS GPUScript;

        public Material ParticleRenderMat;

        void OnRenderObject(){
            DrawObject();
        }

        void DrawObject(){
            Material m = ParticleRenderMat;
            m.SetPass(0);
            m.SetBuffer("_Particles", GPUScript.GetBuffer());
            Graphics.DrawProcedural(MeshTopology.Points, GPUScript.GetMaxParticleNum());
        }

    }

}