"""巨潮资讯网公告连接器。

接入 cninfo.com.cn 的公告查询接口，支持按个股/日期/关键词检索
A 股上市公司公告，获取公告标题、时间、PDF 链接。

用法:
    from app.data.connectors.cninfo import CninfoClient, Announcement

    client = CninfoClient()

    # 按个股查最近公告
    anns = client.fetch_by_stock("000725", days=3)

    # 按日期查全市场
    anns = client.fetch_by_date("2026-05-20", "2026-05-21", plate="szse")

    # 按关键词查
    anns = client.fetch_by_keyword("合作备忘录", days=5)

设计说明:
    - 巨潮是法定信息披露平台，数据权威性最高
    - 使用巨潮官网公开查询接口，无需 API Key
    - 默认 30 秒请求间隔，避免触发反爬
    - 每次请求最多返回 50 条（pageSize 上限）
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ─── 常量 ────────────────────────────────────────────────────────

_BASE_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_PDF_BASE = "http://static.cninfo.com.cn"
_MAX_PAGE_SIZE = 50
_DEFAULT_TIMEOUT = 15  # 秒
_DEFAULT_INTERVAL = 30  # 请求间隔，秒（低频策略无需频繁请求）

# 板块代码
PLATE = {
    "szse": "szse",  # 深市主板
    "szmb": "szmb",  # 深市主板（旧）
    "szzx": "szzx",  # 中小板
    "szcy": "szcy",  # 创业板
    "shmb": "shmb",  # 沪市主板
    "kcb": "kcb",  # 科创板
    "bj": "bj",  # 北交所
}

# 公告分类（财报相关）
CATEGORY = {
    "annual": "category_ndbg_szsh",  # 年度报告
    "semi_annual": "category_bndbg_szsh",  # 半年度报告
    "quarterly": "category_yjdbg_szsh",  # 季度报告 / 业绩预告
    "ipo": "category_scgkfx_szsh",  # 首次公开发行及上市
    "all": "",  # 全部（不筛选）
}

# ─── 数据模型 ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Announcement:
    """上市公司公告。

    所有字段来自巨潮原始数据，不做加工。
    """

    stock_code: str
    """股票代码，如 '000725'。"""

    stock_name: str
    """股票简称，如 '京东方A'。"""

    title: str
    """公告标题。"""

    pub_date: date
    """发布日期。"""

    pub_time: str = ""
    """发布时间，如 '18:20'。"""

    pdf_url: str = ""
    """PDF 下载链接（http://static.cninfo.com.cn/...）。"""

    file_size: int = 0
    """附件大小，单位 KB。"""

    org_id: str = ""
    """机构 ID。"""

    raw_id: str = ""
    """巨潮公告 ID。"""

    def __repr__(self) -> str:
        return (
            f"Announcement({self.stock_code} {self.stock_name} "
            f"{self.pub_date} {self.title[:40]}{'...' if len(self.title) > 40 else ''})"
        )


# ─── 客户端 ──────────────────────────────────────────────────────


class CninfoClient:
    """巨潮资讯网公告查询客户端。

    封装 hisAnnouncement/query 接口，提供按个股、日期、关键词
    三种查询模式。

    属性:
        session: requests.Session（可复用，自带 UA 伪装）
        last_request_time: 上次请求时间（用于间隔控制）
    """

    def __init__(self, interval: float = _DEFAULT_INTERVAL) -> None:
        """
        Args:
            interval: 请求间隔（秒），默认 30 秒。低频策略无需频繁请求。
        """
        self._interval = interval
        self._last_request_at: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
                "Origin": "http://www.cninfo.com.cn",
            }
        )

    # ── 公共查询接口 ─────────────────────────────────────────

    def fetch_by_stock(self, stock_code: str, *, days: int = 3) -> list[Announcement]:
        """查询指定个股最近 N 天的公告。

        Args:
            stock_code: 股票代码，如 '000725'
            days: 查询最近几天，默认 3 天

        Returns:
            公告列表，按发布时间降序排列
        """
        end = date.today()
        start = end - timedelta(days=max(days - 1, 0))
        return self._fetch(
            stock=stock_code,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    def fetch_by_date(
        self,
        start_date: str,
        end_date: str,
        *,
        plate: str = "",
        max_items: int = 200,
    ) -> list[Announcement]:
        """按日期范围查询全市场公告。

        Args:
            start_date: 起始日期，如 '2026-05-20'
            end_date: 结束日期，如 '2026-05-21'
            plate: 板块限制，空=全市场。可选: szse, shmb, szcy, kcb, bj
            max_items: 最多返回条数，默认 200

        Returns:
            公告列表
        """
        return self._fetch(
            start_date=start_date,
            end_date=end_date,
            plate=PLATE.get(plate, plate),
            max_items=max_items,
        )

    def fetch_by_keyword(
        self,
        keyword: str,
        *,
        days: int = 7,
        max_items: int = 100,
    ) -> list[Announcement]:
        """按关键词搜索公告。

        Args:
            keyword: 搜索关键词，如 '合作备忘录'
            days: 搜索最近几天
            max_items: 最多返回条数

        Returns:
            公告列表
        """
        end = date.today()
        start = end - timedelta(days=max(days - 1, 0))
        return self._fetch(
            searchkey=keyword,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            max_items=max_items,
        )

    def fetch_reports(
        self,
        stock_code: str,
        *,
        category: str = "annual",
        years: int = 3,
    ) -> list[Announcement]:
        """查询指定个股的财务报告（年报/半年报/季报）。

        Args:
            stock_code: 股票代码，如 '000725'
            category: 报告类型:
                'annual' — 年度报告
                'semi_annual' — 半年度报告
                'quarterly' — 季度报告/业绩预告
                'all' — 全部不筛选
            years: 回溯年数，默认 3 年

        Returns:
            公告列表（含 PDF 下载链接）

        Example:
            >>> client.fetch_reports("000725", category="annual", years=3)
            [Announcement(000725 京东方A 2025年年度报告, pdf_url=...)]
        """
        cat = CATEGORY.get(category, "")
        end = date.today()
        start = end.replace(year=end.year - years)
        return self._fetch(
            stock=stock_code,
            category=cat,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            max_items=50,
        )

    # ── 内部实现 ─────────────────────────────────────────────

    def _fetch(
        self,
        *,
        stock: str = "",
        searchkey: str = "",
        category: str = "",
        plate: str = "",
        start_date: str = "",
        end_date: str = "",
        max_items: int = 200,
    ) -> list[Announcement]:
        """核心查询方法。

        自动处理分页、间隔控制、重试。

        注意：巨潮 API 的 'stock' 参数实测不生效，个股过滤通过
        searchkey 传股票代码实现（如 searchkey='000725'）。
        """
        results: list[Announcement] = []
        max_pages = (max_items + _MAX_PAGE_SIZE - 1) // _MAX_PAGE_SIZE

        # 个股过滤：通过 searchkey 传股票代码
        effective_searchkey = searchkey or stock

        # 自动推断市场列（用于 column 参数；不传 plate 以覆盖全市场）
        market_column = ""
        if stock and not plate:
            if stock.startswith(("6", "68")):
                market_column = "sse"
            elif stock.startswith(("0", "3", "00", "30")):
                market_column = "szse"

        for page in range(1, max_pages + 1):
            if len(results) >= max_items:
                break

            data = self._request_page(
                page=page,
                searchkey=effective_searchkey,
                category=category,
                plate=plate,
                column=market_column,
                start_date=start_date,
                end_date=end_date,
            )

            if data is None:
                logger.warning("第 %d 页请求失败，停止翻页", page)
                break

            announcements = data.get("announcements")
            if not announcements:
                break  # 没有更多数据

            for item in announcements:
                try:
                    results.append(self._parse_item(item))
                except Exception:
                    logger.debug(
                        "解析公告条目失败: %s", item.get("announcementTitle", "")
                    )
                    continue

            total_pages = data.get("totalpages", 0)
            if page >= total_pages:
                break

            # 间隔控制
            self._wait()

        logger.info(
            "巨潮查询完成: stock=%s keyword=%s date=%s~%s → %d 条",
            stock or "全市场",
            searchkey or "无",
            start_date,
            end_date,
            len(results),
        )
        return results

    def _request_page(
        self,
        *,
        page: int,
        searchkey: str,
        category: str = "",
        plate: str = "",
        column: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> dict | None:
        """请求单页数据。

        Returns:
            JSON 响应字典，失败返回 None
        """
        se_date = f"{start_date}~{end_date}" if start_date and end_date else ""

        payload = {
            "stock": "",
            "searchkey": searchkey,
            "plate": plate,
            "category": category,
            "trade": "",
            "column": column,
            "pageNum": page,
            "pageSize": _MAX_PAGE_SIZE,
            "tabName": "fulltext",
            "sortName": "",
            "sortType": "",
            "limit": "",
            "showTitle": "",
            "seDate": se_date,
        }

        for attempt in range(3):
            try:
                self._wait()
                resp = self._session.post(
                    _BASE_URL,
                    data=payload,
                    timeout=_DEFAULT_TIMEOUT,
                )
                resp.raise_for_status()

                data = resp.json()
                if "announcements" in data:
                    return data

                logger.warning(
                    "巨潮响应无 announcements 字段: %s", list(data.keys())[:5]
                )
                return None

            except requests.Timeout:
                logger.debug("巨潮请求超时 (第 %d 次)", attempt + 1)
            except requests.RequestException as e:
                logger.debug("巨潮请求异常 (第 %d 次): %s", attempt + 1, e)
            except ValueError:
                logger.debug("巨潮 JSON 解析失败 (第 %d 次)", attempt + 1)

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

        return None

    @staticmethod
    def _parse_item(item: dict) -> Announcement:
        """将巨潮 API 返回的条目转换为 Announcement。"""
        # 解析时间
        pub_date = date.today()
        pub_time_str = ""
        ts_raw = item.get("announcementTime")
        if ts_raw:
            try:
                dt = datetime.fromtimestamp(int(ts_raw) / 1000)
                pub_date = dt.date()
                pub_time_str = dt.strftime("%H:%M")
            except (ValueError, OSError):
                pass

        # PDF 链接
        pdf_url = ""
        adjunct = item.get("adjunctUrl", "")
        if adjunct:
            pdf_url = f"{_PDF_BASE}/{adjunct}"

        return Announcement(
            stock_code=str(item.get("secCode", "")),
            stock_name=str(item.get("secName", "")),
            title=str(item.get("announcementTitle", "")),
            pub_date=pub_date,
            pub_time=pub_time_str,
            pdf_url=pdf_url,
            file_size=int(item.get("adjunctSize", 0)),
            org_id=str(item.get("orgId", "")),
            raw_id=str(item.get("id", "")),
        )

    def _wait(self) -> None:
        """确保请求间隔。"""
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request_at = time.monotonic()

    def close(self) -> None:
        """关闭会话。"""
        self._session.close()

    def download_pdf(
        self, announcement: Announcement, save_path: str = ""
    ) -> bytes | None:
        """下载公告 PDF 文件。

        Args:
            announcement: 公告对象（需含有效 pdf_url）
            save_path: 可选保存路径。为空则只返回 bytes，不写文件。

        Returns:
            PDF 二进制内容，失败返回 None
        """
        if not announcement.pdf_url:
            logger.warning("公告无 PDF 链接: %s", announcement)
            return None

        try:
            resp = self._session.get(announcement.pdf_url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("PDF 下载失败: %s — %s", announcement.pdf_url, e)
            return None

        content = resp.content
        if not content.startswith(b"%PDF"):
            logger.warning("下载内容不是有效 PDF: %s", announcement.pdf_url)
            return None

        if save_path:
            import pathlib

            pathlib.Path(save_path).write_bytes(content)
            logger.info("PDF 已保存: %s (%d bytes)", save_path, len(content))

        return content


# ─── 便捷函数 ────────────────────────────────────────────────────

_client: CninfoClient | None = None


def get_client() -> CninfoClient:
    """获取全局单例客户端（复用 Session）。"""
    global _client
    if _client is None:
        _client = CninfoClient()
    return _client


def fetch_stock_announcements(stock_code: str, days: int = 3) -> list[Announcement]:
    """快速查询个股公告（便捷函数）。"""
    return get_client().fetch_by_stock(stock_code, days=days)


def fetch_today_announcements(plate: str = "") -> list[Announcement]:
    """查询今日全市场公告（便捷函数）。"""
    today = date.today().isoformat()
    return get_client().fetch_by_date(today, today, plate=plate)
