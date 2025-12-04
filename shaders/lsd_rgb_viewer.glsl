#uicontrol invlerp normalized(range=[0,1])
#uicontrol float brightness slider(min=0, max=3, default=1.5)
#uicontrol int channelSet slider(min=0, max=12, default=0)

void main() {
  if (channelSet < 3) {
    int baseChannel = channelSet * 3;
    float c1 = getDataValue(baseChannel + 0);
    float c2 = getDataValue(baseChannel + 1);
    float c3 = getDataValue(baseChannel + 2);
    float r = clamp(c1 * brightness, 0.0, 1.0);
    float g = clamp(c2 * brightness, 0.0, 1.0);
    float b = clamp(c3 * brightness, 0.0, 1.0);
    emitRGB(vec3(r, g, b));
  } else {
    int channel = 9 + (channelSet - 3);
    float value = clamp(getDataValue(channel) * brightness, 0.0, 1.0);
    emitGrayscale(value);
  }
}
