#pragma once

// Tracy 可选集成：未定义 IM_FUNCTION_USE_TRACY 时宏为空操作，不依赖 Tracy 库。
#if defined(IM_FUNCTION_USE_TRACY)
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#define ZoneScopedN(name)
#define ZoneScopedC(color)
#define ZoneNamed(varname, name, active)
#define ZoneNamedN(varname, name, color, active)
#define ZoneNamedC(varname, name, color, active)
#define ZoneNamedF(varname, name, color, active)
#define ZoneText(text, size)
#define ZoneTextV(text, size)
#define ZoneTextF(text, ...)
#define ZoneValue(value)
#define ZoneColor(value)
#define FrameMark
#define FrameMarkNamed(name)
#define FrameMarkStart(name)
#define FrameMarkEnd(name)
#define TracyPlot(name, value)
#define TracyPlotConfig(name, type, step, fill, color)
#define TracyMessage(txt, size)
#define TracyMessageL(txt)
#define TracyMessageC(txt, size, color)
#define TracyAlloc(ptr, size)
#define TracyFree(ptr)
#endif
