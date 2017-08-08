using UnityEngine;
using Kodai.NeighborSearch;
using System.Runtime.InteropServices;

// T : Vector2 or Vector3
public abstract class MyParticleSystemBase<Dim, Vec> : MonoBehaviour where Dim : IGridVector<Vec> where Vec : struct {

    [SerializeField] protected ComputeShader ParticleCS;

    [SerializeField] protected ParticleNumEnum numParticleEnum = ParticleNumEnum.NUM_8K;
    [SerializeField] protected int dispIdx;
    [SerializeField] protected Material ParticleRenderMat;

    protected int threadGroupSize;
    protected ComputeBuffer particlesBufferRead;
    protected ComputeBuffer particlesBufferWrite;
    protected static readonly int SIMULATION_BLOCK_SIZE = 32;
    protected int numParticles;

    // If you use grid optimization, use this setting class
    // (I wanted to make this class generics, but Unity can't display generic class on inspector...)
    [SerializeField] private NeighborSearchSetting<Dim, Vec> NSSettings;
    [SerializeField] private Vec particleScatterRange;
    [SerializeField] private Dim gridDimension;
    
    #region Accessor
    public ComputeBuffer ParticlesBufferRead {
        get { return particlesBufferRead; }
    }

    public int NumParticles {
        get { return numParticles; }
    }
    #endregion Accessor
    
    void Start () {
        numParticles = (int)numParticleEnum;
        threadGroupSize = numParticles / SIMULATION_BLOCK_SIZE;

        Debug.Log(NumParticles);

        particlesBufferRead = new ComputeBuffer(numParticles, Marshal.SizeOf(typeof(MyParticleData<Vec>)));
        particlesBufferWrite = new ComputeBuffer(numParticles, Marshal.SizeOf(typeof(MyParticleData<Vec>)));

        // unityのインスペクタがジェネリクスに対応していないので冗長な実装になっている
        NSSettings = new NeighborSearchSetting<Dim, Vec>(numParticleEnum, particleScatterRange, gridDimension);

        particlesBufferRead.SetData(ScatterParticle(numParticles, NSSettings.particleScatterRange));
    }
	
	// Update is called once per frame
	void Update () {
        // ---- Grid Optimization -------------------------------------------------------------------
        NSSettings.GridOptimizer.GridSort(ref particlesBufferRead);    // Pass the buffer you want to optimize

        // ---- Your Particle Process ---------------------------------------------------------------
        ParticleCS.SetInt("_NumParticles", numParticles);
        ParticleCS.SetVector("_GridDim", NSSettings.gridDimension.ToVector3());
        ParticleCS.SetInt("_DispIdx", dispIdx);
        ParticleCS.SetFloat("_GridH", NSSettings.GridOptimizer.GridH);

        int kernel = ParticleCS.FindKernel("Update");
        ParticleCS.SetBuffer(kernel, "_ParticlesBufferRead", particlesBufferRead);
        ParticleCS.SetBuffer(kernel, "_ParticlesBufferWrite", particlesBufferWrite);
        ParticleCS.SetBuffer(kernel, "_GridIndicesBufferRead", NSSettings.GridOptimizer.GridIndicesBuffer);   // Get and use a GridIndicesBuffer to find neighbor
        ParticleCS.Dispatch(kernel, threadGroupSize, 1, 1);

        SwapBuffer(ref particlesBufferRead, ref particlesBufferWrite);
    }

    private void OnRenderObject() {
        Material m = ParticleRenderMat;
        m.SetPass(0);
        m.SetBuffer("_Particles", ParticlesBufferRead);
        Graphics.DrawProcedural(MeshTopology.Points, NumParticles);
    }

    void OnDestroy() {
        DestroyBuffer(particlesBufferRead);
        DestroyBuffer(particlesBufferWrite);
        NSSettings.GridOptimizer.Dispose(); // Must
    }

    void SwapBuffer(ref ComputeBuffer src, ref ComputeBuffer dst) {
        ComputeBuffer tmp = src;
        src = dst;
        dst = tmp;
    }

    void DestroyBuffer(ComputeBuffer buffer) {
        if (buffer != null) {
            buffer.Release();
            buffer = null;
        }
    }

    protected abstract MyParticleData<Vec>[] ScatterParticle(int num, Vec range);
}
