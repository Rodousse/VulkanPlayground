#version 450
#extension GL_ARB_separate_shader_objects : enable 

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 vertexPos;
layout(location = 3) out vec3 lightDir;
layout(location = 4) out vec3 camDir;

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
    vec3 lightPos;
}ubo;

out gl_PerVertex{
    vec4 gl_Position;
};

void main()
{
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);

    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(ubo.model))); //Prevent Normal deformation from non uniform model matrice

    lightDir = normalize(ubo.lightPos - worldPos.xyz);
    camDir = (inverse(ubo.view)*vec4(0.0,0.0,0.0,1.0) - worldPos).xyz;
    fragNormal = normalize(normalMatrix * inNormal);
    fragTexCoord = inTexCoord;
}
