import pyautogui
import time
import Quartz
import cv2
import numpy as np
from PIL import Image, ImageDraw
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mhxy")

pyautogui.FAILSAFE = False

WINDOW_TITLE = "Parallels Desktop"
CLICK_OFFSET_X = 0   # 全局点击偏移，用于补偿游标与实际点击位置的误差
CLICK_OFFSET_Y = 0
_SCALE_RANGE = np.arange(0.3, 1.6, 0.05)  # 多尺度匹配的缩放区间


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    win = get_target_window()
    if not win:
        log.warning("⚠️  未找到窗口: %s", WINDOW_TITLE)
        exit(0)
    log.info("🖥  窗口 %s  (%d, %d)  %dx%d", WINDOW_TITLE, win["left"], win["top"], win["width"], win["height"])

    # 右键点击英雄，触发上下文菜单
    match_hero = find_in_window(win, "images/hero.png", debug=True, debug_name="debug_match_hero.png")
    if not match_hero:
        log.warning("⚠️  未找到英雄")
        return False
    pyautogui.rightClick(*match_hero)

    # 等待「云游道人」出现后点击，再确认购买宝贝
    while True:
        if find_and_click(win, "images/yydr.png", confidence=0.5, debug=True, debug_name="debug_match_yydr.png"):
            time.sleep(0.3)
            if find_and_click(win, "images/gmbb.png", confidence=0.7, debug=True, debug_name="debug_match_gmbb.png"):
                break
            log.warning("⚠️  未找到购买宝贝")
        else:
            log.warning("⚠️  未找到「云游道人」")
            time.sleep(1)

    time.sleep(0.2)
    for _ in range(3):
        find_and_click(win, "images/gyms_gj.png", debug=True, debug_name="debug_match_gyms_gj.png", confidence=0.6)
        time.sleep(0.1)
        find_and_click(win, "images/queding.png", debug=True, debug_name="debug_match_queding.png", confidence=0.7)
        time.sleep(0.2)


# ─── 窗口 ──────────────────────────────────────────────────────────────────────

def get_target_window():
    """遍历屏幕上所有窗口，返回匹配 WINDOW_TITLE 且尺寸合法的第一个窗口信息。"""
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for win in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID):
        name = win.get("kCGWindowName", "") or ""
        owner = win.get("kCGWindowOwnerName", "") or ""
        if WINDOW_TITLE not in owner and WINDOW_TITLE not in name:
            continue
        bounds = win.get("kCGWindowBounds")
        if not bounds:
            continue
        w, h = int(bounds["Width"]), int(bounds["Height"])
        if w >= 100 and h >= 100:
            return {"left": int(bounds["X"]), "top": int(bounds["Y"]), "width": w, "height": h}
    return None


# ─── 模板匹配 ──────────────────────────────────────────────────────────────────

def find_in_window(win, template_path, confidence=0.7, debug=False, debug_name="debug_match.png", search_region=None):
    """在窗口截图中用多尺度模板匹配定位目标，返回屏幕坐标 (x, y) 或 None。"""
    screenshot = pyautogui.screenshot(region=(win["left"], win["top"], win["width"], win["height"]))
    # Retina/缩放屏下截图像素与窗口逻辑坐标不等比，scale 用于两者互转
    scale = screenshot.width / win["width"]
    haystack = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    needle_base = cv2.cvtColor(np.array(Image.open(template_path)), cv2.COLOR_RGB2BGR)

    name = _tname(template_path)
    roi_w = _win_roi(win, search_region)
    if roi_w is None:
        log.warning("⚠️  %-12s  搜索区域无效", name)
        return None

    roi_s = _shot_roi(roi_w, scale, haystack.shape)
    if roi_s is None:
        log.warning("⚠️  %-12s  截图搜索区域无效", name)
        return None

    ls, ts, rs, bs = roi_s
    best_val, roi_loc, best_needle = _multiscale_match(haystack[ts:bs, ls:rs], needle_base)

    if roi_loc is None:
        log.warning("⚠️  %-12s  ROI内未找到可用匹配", name)
        return None

    # roi_loc 是 ROI 内的局部坐标，加上 ROI 偏移才是截图上的绝对坐标
    best_loc = (roi_loc[0] + ls, roi_loc[1] + ts)

    if debug:
        # 黄框标注搜索区域，红框标注匹配结果
        _save_debug(screenshot, roi_s if search_region else None, best_loc, best_needle, debug_name)

    if best_val < confidence:
        log.info("❌ %-12s  置信度 %.3f < %.1f  未匹配", name, best_val, confidence)
        return None
    log.info("✅ %-12s  置信度 %.3f", name, best_val)

    # 取匹配框中心，再从截图像素坐标换算回屏幕逻辑坐标
    cx = best_loc[0] + best_needle.shape[1] / 2
    cy = best_loc[1] + best_needle.shape[0] / 2
    return win["left"] + cx / scale, win["top"] + cy / scale


def _multiscale_match(haystack_roi, needle_base):
    """在 _SCALE_RANGE 范围内逐比例缩放模板，返回置信度最高的 (val, loc, needle)。"""
    best_val, best_loc, best_needle = -1.0, None, needle_base
    for s in _SCALE_RANGE:
        w = int(needle_base.shape[1] * s)
        h = int(needle_base.shape[0] * s)
        # 模板超出 ROI 或过小时跳过，避免无效匹配
        if w < 10 or h < 10 or w > haystack_roi.shape[1] or h > haystack_roi.shape[0]:
            continue
        needle = cv2.resize(needle_base, (w, h))
        _, val, _, loc = cv2.minMaxLoc(cv2.matchTemplate(haystack_roi, needle, cv2.TM_CCOEFF_NORMED))
        if val > best_val:
            best_val, best_loc, best_needle = val, loc, needle
    return best_val, best_loc, best_needle


def _win_roi(win, search_region):
    """将 search_region 的屏幕绝对坐标转换为窗口相对坐标，返回 (l, t, r, b) 或 None。"""
    l, t, r, b = 0, 0, win["width"], win["height"]
    if search_region:
        if (v := search_region.get("x_min")) is not None:
            l = max(l, int(v - win["left"]))
        if (v := search_region.get("x_max")) is not None:
            r = min(r, int(v - win["left"]))
        if (v := search_region.get("y_min")) is not None:
            t = max(t, int(v - win["top"]))
        if (v := search_region.get("y_max")) is not None:
            b = min(b, int(v - win["top"]))
    return (l, t, r, b) if l < r and t < b else None


def _shot_roi(win_roi, scale, h_shape):
    """将窗口相对坐标 ROI 按 scale 映射到截图像素坐标，返回 (l, t, r, b) 或 None。"""
    l, t, r, b = win_roi
    px = lambda v: int(round(v * scale))
    ls, ts = max(0, px(l)), max(0, px(t))
    rs, bs = min(h_shape[1], px(r)), min(h_shape[0], px(b))
    return (ls, ts, rs, bs) if ls < rs and ts < bs else None


def _save_debug(screenshot, roi_shot, best_loc, best_needle, debug_name):
    """将搜索区域（黄框）和匹配结果（红框）绘制到截图上并保存。"""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)
    if roi_shot:
        draw.rectangle(list(roi_shot), outline="yellow", width=2)
    x1, y1 = best_loc
    draw.rectangle([x1, y1, x1 + best_needle.shape[1], y1 + best_needle.shape[0]], outline="red", width=3)
    img.save(debug_name)


def _tname(path: str) -> str:
    """从路径中提取不含扩展名的文件名，用于日志展示。"""
    return path.split("/")[-1].replace(".png", "")


# ─── 点击 ──────────────────────────────────────────────────────────────────────

def click_match(screen_x, screen_y, click_count=1):
    """在指定屏幕坐标点击，支持多次点击。"""
    x = screen_x - CLICK_OFFSET_X
    y = screen_y - CLICK_OFFSET_Y
    for i in range(click_count):
        pyautogui.click(x, y)
        log.info("🖱  点击 (%.0f, %.0f)  第 %d/%d 次", x, y, i + 1, click_count)
        time.sleep(0.1)
    return True


def find_and_click(win, template_path, confidence=0.7, debug=False, debug_name="debug_match.png", click_count=1):
    """找到模板后立即点击；每次调用都重新获取窗口位置，防止窗口移动后坐标漂移。"""
    current_win = get_target_window() or win
    if not current_win:
        log.warning("⚠️  未找到窗口: %s", WINDOW_TITLE)
        return False
    match = find_in_window(current_win, template_path, confidence=confidence, debug=debug, debug_name=debug_name)
    if not match:
        return False
    return click_match(*match, click_count=click_count)


# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
