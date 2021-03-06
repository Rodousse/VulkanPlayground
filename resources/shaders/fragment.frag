#version 450
#extension GL_ARB_separate_shader_objects : enable 

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 vertexPos;
layout(location = 3) in vec3 lightDir;
layout(location = 4) in vec3 camDir;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 albedo = vec3(1.0);

    vec3 N = normalize(fragNormal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(camDir);
    vec3 R = reflect(L,N);
    float alpha = 100.0;

    vec3 diff = max(dot(N,L), 0.0) * albedo;
    vec3 spec = pow(max(dot(R,V), 0.0), alpha) * vec3(0.5);
    outColor = vec4(diff + spec + (albedo * 0.05), 1.0);
}
