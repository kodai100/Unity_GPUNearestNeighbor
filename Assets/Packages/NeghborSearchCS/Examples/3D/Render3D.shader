Shader "Custom/Render3D" {
	Properties{
		_MainTex("Texture", 2D) = "white" {}
		_ParticleRad("ParticleRadius", Range(0.01, 1)) = 0.05
	}

	SubShader{
		ZWrite On
		Blend SrcAlpha OneMinusSrcAlpha

		Pass{
			CGPROGRAM

			#pragma target 5.0
			#pragma vertex vert
			#pragma geometry geom
			#pragma fragment frag

			#include "UnityCG.cginc"

			sampler2D _MainTex;
			float _ParticleRad;

			struct Particle {
				float3 pos;
				float3 color;
			};

			StructuredBuffer<Particle> _Particles;

			struct v2g {
				float4 pos : SV_POSITION;
				float2 tex : TEXCOORD0;
				float4 col : COLOR;
			};

			v2g vert(uint id : SV_VertexID) {
				v2g output;
				output.pos = float4(_Particles[id].pos, 1);
				output.tex = float2(0, 0);
				output.col = float4(_Particles[id].color, 1);
				return output;
			}

			[maxvertexcount(4)]
			void geom(point v2g input[1], inout TriangleStream<v2g> outStream) {
				v2g output;

				float4 pos = input[0].pos;
				float4 col = input[0].col;

				for (int x = 0; x < 2; x++) {
					for (int y = 0; y < 2; y++) {
						float4x4 billboardMatrix = UNITY_MATRIX_V;
						billboardMatrix._m03 =
							billboardMatrix._m13 =
							billboardMatrix._m23 =
							billboardMatrix._m33 = 0;

						float2 tex = float2(x, y);
						output.tex = tex;

						output.pos = pos + mul(float4((tex * 2 - float2(1, 1)) * _ParticleRad, 0, 1), billboardMatrix);
						output.pos = mul(UNITY_MATRIX_VP, output.pos);

						output.col = col;

						outStream.Append(output);
					}
				}

				outStream.RestartStrip();
			}

			fixed4 frag(v2g i) : COLOR{

				fixed4 color = tex2D(_MainTex, i.tex) * i.col;

				if (color.a < 0.8) discard;
					 
				return color;
			}

			ENDCG
		}
	}
}