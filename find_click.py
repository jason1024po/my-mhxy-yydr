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
CLICK_OFFSET_X = 0
CLICK_OFFSET_Y = 0

def get_target_window():
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    win_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    for win in win_list:
        name = win.get("kCGWindowName", "") or ""
        owner = win.get("kCGWindowOwnerName", "") or ""
        if WINDOW_TITLE in owner or WINDOW_TITLE in name:
            bounds = win.get("kCGWindowBounds")
            if not bounds:
                continue
            w = int(bounds["Width"])
            h = int(bounds["Height"])
            if w < 100 or h < 100:
                continue
            return {"left": int(bounds["X"]), "top": int(bounds["Y"]), "width": w, "height": h}
    return None

def find_in_window(win, template_path, confidence=0.7, debug=False, debug_name="debug_match.png", search_region=None):
    # 先按窗口截图，后续所有匹配都在这张图上进行。
    screenshot = pyautogui.screenshot(region=(win["left"], win["top"], win["width"], win["height"]))
    # Retina/缩放场景下，截图像素和窗口坐标可能不一致，用 scale 做坐标换算。
    scale = screenshot.width / win["width"]
    haystack = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    needle_base = cv2.cvtColor(np.array(Image.open(template_path)), cv2.COLOR_RGB2BGR)

    # ROI 默认是整窗；如果传入 search_region，则收缩到指定搜索范围。
    roi_left_win = 0
    roi_top_win = 0
    roi_right_win = win["width"]
    roi_bottom_win = win["height"]
    if search_region:
        x_min = search_region.get("x_min")
        x_max = search_region.get("x_max")
        y_min = search_region.get("y_min")
        y_max = search_region.get("y_max")
        if x_min is not None:
            roi_left_win = max(roi_left_win, int(x_min - win["left"]))
        if x_max is not None:
            roi_right_win = min(roi_right_win, int(x_max - win["left"]))
        if y_min is not None:
            roi_top_win = max(roi_top_win, int(y_min - win["top"]))
        if y_max is not None:
            roi_bottom_win = min(roi_bottom_win, int(y_max - win["top"]))

    # 先在窗口坐标系做有效性校验，避免出现空区域。
    if roi_left_win >= roi_right_win or roi_top_win >= roi_bottom_win:
        log.warning("⚠️  %-12s  搜索区域无效", template_path.split("/")[-1])
        return None

    # 将窗口坐标系 ROI 映射到截图像素坐标系。
    roi_left_shot = max(0, int(round(roi_left_win * scale)))
    roi_top_shot = max(0, int(round(roi_top_win * scale)))
    roi_right_shot = min(haystack.shape[1], int(round(roi_right_win * scale)))
    roi_bottom_shot = min(haystack.shape[0], int(round(roi_bottom_win * scale)))
    if roi_left_shot >= roi_right_shot or roi_top_shot >= roi_bottom_shot:
        log.warning("⚠️  %-12s  截图搜索区域无效", template_path.split("/")[-1])
        return None

    # 真正参与模板匹配的搜索图。
    haystack_roi = haystack[roi_top_shot:roi_bottom_shot, roi_left_shot:roi_right_shot]

    best_val = -1
    best_loc = None
    best_needle = needle_base

    # 多尺度匹配：遍历模板缩放比例，取置信度最高的一次。
    for s in np.arange(0.3, 1.6, 0.05):
        w = int(needle_base.shape[1] * s)
        h = int(needle_base.shape[0] * s)
        if w < 10 or h < 10 or w > haystack_roi.shape[1] or h > haystack_roi.shape[0]:
            continue
        needle = cv2.resize(needle_base, (w, h))
        result = cv2.matchTemplate(haystack_roi, needle, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(result)
        if val > best_val:
            best_val = val
            # loc 是 ROI 内坐标，转换回整张截图坐标，便于后续统一处理。
            best_loc = (loc[0] + roi_left_shot, loc[1] + roi_top_shot)
            best_needle = needle

    if best_loc is None:
        log.warning("⚠️  %-12s  ROI内未找到可用匹配", template_path.split("/")[-1])
        return None

    if debug:
        # debug 图里：黄框是 ROI，红框是最终匹配框。
        debug_img = screenshot.copy()
        draw = ImageDraw.Draw(debug_img)
        if search_region:
            draw.rectangle([roi_left_shot, roi_top_shot, roi_right_shot, roi_bottom_shot], outline="yellow", width=2)
        x1, y1 = best_loc
        x2 = x1 + best_needle.shape[1]
        y2 = y1 + best_needle.shape[0]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        debug_img.save(debug_name)

    name = template_path.split("/")[-1].replace(".png", "")
    if best_val < confidence:
        log.info("❌ %-12s  置信度 %.3f < %.1f  未匹配", name, best_val, confidence)
        return None
    log.info("✅ %-12s  置信度 %.3f", name, best_val)

    # 匹配框中心点（截图坐标）再映射回屏幕坐标，供 pyautogui 点击。
    x1, y1 = best_loc
    cx_in_shot = x1 + best_needle.shape[1] / 2
    cy_in_shot = y1 + best_needle.shape[0] / 2
    screen_x = win["left"] + cx_in_shot / scale
    screen_y = win["top"] + cy_in_shot / scale
    return screen_x, screen_y


def click_match(screen_x, screen_y, move_duration=0.2, click_count=1):
    target_x = screen_x - CLICK_OFFSET_X
    target_y = screen_y - CLICK_OFFSET_Y
    # pyautogui.moveTo(target_x, target_y, duration=move_duration)
    for i in range(click_count):
        pyautogui.click(target_x, target_y)
        log.info("🖱  点击 (%.0f, %.0f)  第 %d/%d 次", target_x, target_y, i + 1, click_count)
        time.sleep(0.1)
    return True


def find_and_click(win, template_path, confidence=0.7, debug=False, debug_name="debug_match.png", move_duration=0.3, click_count=1):
    # 动态刷新窗口范围，避免游戏切换后坐标漂移
    current_win = get_target_window() or win
    if not current_win:
        log.warning("⚠️  未找到窗口: %s", WINDOW_TITLE)
        return False
    match = find_in_window(current_win, template_path, confidence=confidence, debug=debug, debug_name=debug_name)
    if not match:
        return False
    screen_x, screen_y = match
    return click_match(screen_x, screen_y, move_duration=move_duration, click_count=click_count)

def main():
    global CLICK_OFFSET_X, CLICK_OFFSET_Y
    win = get_target_window()
    if not win:
        log.warning("⚠️  未找到窗口: %s", WINDOW_TITLE)
        exit(0)
    log.info("🖥  窗口 %s  (%d, %d)  %dx%d", WINDOW_TITLE, win["left"], win["top"], win["width"], win["height"])

    match_hero = find_in_window(win, "images/hero.png", debug=True, debug_name="debug_match_hero.png")

    if not match_hero:
        log.warning("⚠️  未找到英雄")
        return False
    hero_screen_x, hero_screen_y = match_hero
    pyautogui.rightClick(hero_screen_x, hero_screen_y, duration=0.1)
    
    # mouse_search_region = {
    #     "x_min": hero_screen_x,
    #     "y_max": 2 * hero_screen_y,
    # }
    # match_mouse = find_in_window(
    #     win,
    #     "images/mouse.png",
    #     confidence=0.5,
    #     debug=True,
    #     debug_name="debug_match_mouse.png",
    #     search_region=mouse_search_region,
    # )

    # if not match_mouse:
    #     log.warning("⚠️  未找到鼠标")
    #     return False
    # mouse_screen_x, mouse_screen_y = match_mouse

    # CLICK_OFFSET_X = int(round(mouse_screen_x - hero_screen_x))
    # CLICK_OFFSET_Y = int(round(mouse_screen_y - hero_screen_y))
    log.info("📌 全局偏移已更新: x=%d, y=%d", CLICK_OFFSET_X, CLICK_OFFSET_Y)


    
    while True:
        if find_and_click(win, "images/yydr.png", confidence=0.5, move_duration=0.4, debug=True, debug_name="debug_match_yydr.png"):
            time.sleep(0.2)
            if not find_and_click(win, "images/gmbb.png", confidence=0.7,  debug=True, debug_name="debug_match_gmbb.png"):
                log.warning("⚠️  未找到购买宝贝")
                continue
            else:
                break
        else:
            log.warning("⚠️  未找到「云游道人」")
            time.sleep(1)
            continue
            
    time.sleep(0.2)
    find_and_click(win, "images/gyms_gj.png", debug=True, debug_name="debug_match_gyms_gj.png", confidence=0.6)
    time.sleep(0.1)
    find_and_click(win, "images/queding.png", debug=True, debug_name="debug_match_queding.png", confidence=0.7)
    

if __name__ == "__main__":
    main()
