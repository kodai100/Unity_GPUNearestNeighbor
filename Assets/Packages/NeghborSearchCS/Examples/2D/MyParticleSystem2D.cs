using UnityEngine;
using Kodai.NeighborSearch;

/// <summary>
/// Particle system with Grid Optimization
/// </summary>
public class MyParticleSystem2D : MyParticleSystemBase<IntDim2D, Vector2> {

    protected override MyParticleData<Vector2>[] ScatterParticle(int num, Vector2 range) {
        var particles = new MyParticleData<Vector2>[num];
        for (int i = 0; i < num; i++) {
            particles[i] = new MyParticleData<Vector2>(new Vector2(Random.Range(1, range.x), Random.Range(1, range.y)));
        }
        return particles;
    }
}