
// ---------------------
// Define Data structure (must be same as your particle data)
// ---------------------
struct Data {
	float3 pos;
	float3 color;
};

cbuffer grid {
	float3 _GridDim;
	float _GridH;
};

StructuredBuffer  <uint2>	_GridIndicesBufferRead;
RWStructuredBuffer<uint2>	_GridIndicesBufferWrite;


// 所属するセルの2次元インデックスを返す
float3 GridCalculateCell(float3 pos) {
	return pos / _GridH;
}

// セルの2次元インデックスから1次元インデックスを返す
uint GridKey(uint3 xyz) {
	return xyz.x + xyz.y * _GridDim.x + xyz.z * _GridDim.x * _GridDim.y;
}

// (グリッドID, パーティクルID) のペアを作成する
uint2 MakeKeyValuePair(uint3 xyz, uint value) {
	// uint2([GridHash], [ParticleID]) 
	return uint2(GridKey(xyz), value);	// 逆?
}

// グリッドIDとパーティクルIDのペアからグリッドIDだけを抜き出す
uint GridGetKey(uint2 pair) {
	return pair.x;
}

// グリッドIDとパーティクルIDのペアからパーティクルIDだけを抜き出す
uint GridGetValue(uint2 pair) {
	return pair.y;
}

#define LOOP_AROUND_NEIGHBOR(pos) int3 G_XYZ = (int3)GridCalculateCell(pos); for(int Z = max(G_XYZ.z - 1, 0); Z <= min(G_XYZ.z + 1, _GridDim.z - 1); Z++) for (int Y = max(G_XYZ.y - 1, 0); Y <= min(G_XYZ.y + 1, _GridDim.y - 1); Y++)  for (int X = max(G_XYZ.x - 1, 0); X <= min(G_XYZ.x + 1, _GridDim.x - 1); X++)