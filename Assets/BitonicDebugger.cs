using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;

public class BitonicDebugger : MonoBehaviour {
    public ComputeShader BitonicSortShader;

    const int BUFFER_SIZE = 2048;

    const uint BITONIC_BLOCK_SIZE = 512;
    const uint TRANSPOSE_BLOCK_SIZE = 16;

    const int KERNEL_ID_BITONICSORT = 0;
    const int KERNEL_ID_TRANSPOSE_MATRIX = 1;

    ComputeBuffer _inBuffer;
    ComputeBuffer _tempBuffer;

    void Start() {
        Debug.Log("<color=lime>1 key :Sort, 2 key : Reset</color>");

        Init();
    }

    void Update() {
        if (Input.GetKeyUp("1")) {
            // Sort
            GPUSort(_inBuffer, _tempBuffer);
            ShowValuesOnConsole(_inBuffer, "sorted : ");
        }

        if (Input.GetKeyUp("2")) {
            // Reset
            Reset();
        }
    }

    void Init() {
        _inBuffer = new ComputeBuffer(BUFFER_SIZE, Marshal.SizeOf(typeof(Uint2)));
        _tempBuffer = new ComputeBuffer(BUFFER_SIZE, Marshal.SizeOf(typeof(Uint2)));

        Reset();
    }

    void Reset() {
        Uint2[] data = new Uint2[BUFFER_SIZE];
        for (var i = 0; i < data.Length; i++) {
            Uint2 tmp; tmp.x = (uint)(Random.Range(0, BUFFER_SIZE)); tmp.y = (uint)i;
            data[i] = tmp;
        }

        _inBuffer.SetData(data);
        _tempBuffer.SetData(data);

        ShowValuesOnConsole(_inBuffer, "No sort : ");
    }

    void GPUSort(ComputeBuffer inBuffer, ComputeBuffer tempBuffer) {
        ComputeShader shader = BitonicSortShader;
        // Determine parameters.
        uint NUM_ELEMENTS = (uint)BUFFER_SIZE;
        uint MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
        uint MATRIX_HEIGHT = (uint)NUM_ELEMENTS / BITONIC_BLOCK_SIZE;

        // Sort the data
        // First sort the rows for the levels <= to the block size
        for (uint level = 2; level <= BITONIC_BLOCK_SIZE; level <<= 1) {
            SetGPUSortConstants(shader, level, level, MATRIX_HEIGHT, MATRIX_WIDTH);

            // Sort the row data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }

        // Then sort the rows and columns for the levels > than the block size
        // Transpose. Sort the Columns. Transpose. Sort the Rows.
        for (uint level = (BITONIC_BLOCK_SIZE << 1); level <= NUM_ELEMENTS; level <<= 1) {
            // Transpose the data from buffer 1 into buffer 2
            SetGPUSortConstants(shader, (level / BITONIC_BLOCK_SIZE), (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input", inBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data", tempBuffer);
            shader.Dispatch(KERNEL_ID_TRANSPOSE_MATRIX, (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the transposed column data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data", tempBuffer);
            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);

            // Transpose the data from buffer 2 back into buffer 1
            SetGPUSortConstants(shader, BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Input", tempBuffer);
            shader.SetBuffer(KERNEL_ID_TRANSPOSE_MATRIX, "Data", inBuffer);
            shader.Dispatch(KERNEL_ID_TRANSPOSE_MATRIX, (int)(MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE), (int)(MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE), 1);

            // Sort the row data
            shader.SetBuffer(KERNEL_ID_BITONICSORT, "Data", inBuffer);
            shader.Dispatch(KERNEL_ID_BITONICSORT, (int)(NUM_ELEMENTS / BITONIC_BLOCK_SIZE), 1, 1);
        }
    }

    void SetGPUSortConstants(ComputeShader cs, uint level, uint levelMask, uint width, uint height) {
        cs.SetInt("_Level", (int)level);
        cs.SetInt("_LevelMask", (int)levelMask);
        cs.SetInt("_Width", (int)width);
        cs.SetInt("_Height", (int)height);
    }

    void ShowValuesOnConsole(ComputeBuffer buffer, string label) {
        if (buffer == null || buffer.count == 0) return;
        var values = "";
        var data = new Uint2[buffer.count];
        buffer.GetData(data);
        for (var i = 0; i < data.Length; i++) {
            values += data[i].x + " ";
        }
        Debug.Log(label + values);
    }

    void OnDestroy() {
        if (_inBuffer != null) {
            _inBuffer.Release();
        }
        _inBuffer = null;

        if (_tempBuffer != null) {
            _tempBuffer.Release();
        }
        _tempBuffer = null;
    }
}
