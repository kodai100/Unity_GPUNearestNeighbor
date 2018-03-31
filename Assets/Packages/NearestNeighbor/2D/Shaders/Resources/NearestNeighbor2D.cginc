struct Data {
	float2 pos;
	float3 color;
};

cbuffer grid {
	float2 _GridDim;
	float _GridH;
};

StructuredBuffer  <uint2> _GridIndicesBufferRead;
RWStructuredBuffer<uint2> _GridIndicesBufferWrite;


float2 GridCalculateCell(float2 pos) {
	return pos / _GridH;
}

uint GridKey(uint2 xy) {
	return xy.x + xy.y * _GridDim.x;
}

uint2 MakeKeyValuePair(uint2 xy, uint value) {
	return uint2(GridKey(xy), value);
}

uint GridGetKey(uint2 pair) {
	return pair.x;
}

uint GridGetValue(uint2 pair) {
	return pair.y;
}

#define LOOP_AROUND_NEIGHBOR(pos) int2 G_XY = (int2)GridCalculateCell(pos); for (int Y = max(G_XY.y - 1, 0); Y <= min(G_XY.y + 1, _GridDim.y - 1); Y++) for (int X = max(G_XY.x - 1, 0); X <= min(G_XY.x + 1, _GridDim.x - 1); X++)