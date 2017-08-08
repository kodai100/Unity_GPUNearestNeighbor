using System.Runtime.InteropServices;
using UnityEngine;
using Kodai.NeighborSearch;

/// <summary>
/// Particle system with Grid Optimization
/// </summary>
public class MyParticleSystem3D : MyParticleSystemBase<IntDim3D, Vector3> {
    
    protected override MyParticleData<Vector3>[] ScatterParticle(int num, Vector3 range) {

        var particles = new MyParticleData<Vector3>[num];
        for (int i = 0; i < num; i++) {
            particles[i] = new MyParticleData<Vector3>(new Vector3(Random.Range(1, range.x), Random.Range(1, range.y), Random.Range(1, range.z)));
        }
        return particles;
    }
    
}



