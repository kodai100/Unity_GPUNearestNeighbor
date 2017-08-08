using UnityEngine;

/// <summary>
/// Define your particle data 
/// </summary>
public struct MyParticleData<T> where T : struct{
    public T pos;
    public Vector3 color;

    public MyParticleData(T pos) {
        this.pos = pos;
        this.color = new Vector3(1, 1, 1);
    }
}