from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    JavascriptException,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
import random
import time
import json
import os
import logging


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper_search.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass
class ScraperConfig:
    usernames: list[str]
    date_from: datetime
    date_to: datetime
    chrome_profile_path: str
    output_folder: str = "data_raw"

    # Tamaño de cada tramo de búsqueda (en días)
    # Tramos más pequeños = menos tweets por búsqueda = más fiable
    # Recomendado: 30-60 días. Si un tramo da pocos resultados, prueba 14.
    search_window_days: int = 45

    # Parada por estancamiento dentro de un tramo
    max_rounds_without_new_ids: int = 6
    max_stalled_oldest_rounds: int = 4

    # Pausas (segundos)
    scroll_pause_min: float = 3.0
    scroll_pause_max: float = 6.5
    pause_between_windows_min: float = 8.0
    pause_between_windows_max: float = 18.0
    pause_between_users_min: float = 40.0
    pause_between_users_max: float = 75.0

    # Reintentos
    max_retries: int = 3

    # Guardado incremental
    save_every_n_tweets: int = 10

    fetch_context_for_replies: bool = False


CONFIG = ScraperConfig(
    usernames=["NunezFeijoo", "Santi_ABASCAL", "sanchezcastejon", "Yolanda_Diaz_"],
    date_from=datetime(2022, 4, 23, tzinfo=timezone.utc),
    date_to=datetime(2026, 4, 23, tzinfo=timezone.utc),
    chrome_profile_path=r"C:\Users\narci\chrome_selenium_profile",
)

os.makedirs(CONFIG.output_folder, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def human_sleep(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))


def generate_date_windows(
    date_from: datetime, date_to: datetime, window_days: int
) -> list[tuple[datetime, datetime]]:
    """
    Divide el rango total en ventanas de `window_days` días.
    Devuelve la lista ordenada de MÁS RECIENTE a MÁS ANTIGUA,
    igual que haría un scroll natural en X.
    """
    windows = []
    current_end = date_to
    while current_end > date_from:
        current_start = max(current_end - timedelta(days=window_days), date_from)
        windows.append((current_start, current_end))
        current_end = current_start

    # Más reciente primero
    return windows


def build_search_url(username: str, since: datetime, until: datetime) -> str:
    """
    Construye la URL de búsqueda avanzada de X con filtro de autor y fechas.
    Usa el modo 'live' (últimos/cronológico) para máxima cobertura.
    """
    since_str = since.strftime("%Y-%m-%d")
    until_str = until.strftime("%Y-%m-%d")
    query = f"from:{username} since:{since_str} until:{until_str}"
    encoded = quote(query)
    return f"https://x.com/search?q={encoded}&src=typed_query&f=live"


# -----------------------------
# Scraper Class
# -----------------------------
class XSearchScraper:
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.driver = self._init_driver(config.chrome_profile_path)
        self.wait = WebDriverWait(self.driver, 20)

    def _init_driver(self, profile_path: str) -> webdriver.Chrome:
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """
            },
        )
        return driver

    def quit(self):
        self.driver.quit()

    # -----------------------------
    # Scroll helpers
    # -----------------------------
    def human_scroll(self):
        total_px = random.randint(600, 1200)
        steps = random.randint(3, 6)
        per_step = total_px // steps
        for _ in range(steps):
            self.driver.execute_script("window.scrollBy(0, arguments[0]);", per_step)
            time.sleep(random.uniform(0.08, 0.35))

    def scroll_to_last_article(self) -> bool:
        try:
            articles = self.driver.find_elements(By.TAG_NAME, "article")
            if not articles:
                return False
            last_article = articles[-1]
            self.driver.execute_script(
                """
                arguments[0].scrollIntoView({
                    behavior: 'instant',
                    block: 'end',
                    inline: 'nearest'
                });
                """,
                last_article,
            )
            time.sleep(random.uniform(0.4, 1.1))
            extra_px = random.randint(250, 700)
            self.driver.execute_script("window.scrollBy(0, arguments[0]);", extra_px)
            time.sleep(random.uniform(0.2, 0.8))
            return True
        except (StaleElementReferenceException, JavascriptException):
            return False

    # -----------------------------
    # URL helpers
    # -----------------------------
    @staticmethod
    def normalize_url(url: str) -> str | None:
        if not url:
            return None
        return url.split("?")[0].rstrip("/")

    def get_status_links(self, article) -> list[str]:
        links = article.find_elements(By.TAG_NAME, "a")
        seen = []
        for link in links:
            href = self.normalize_url(link.get_attribute("href"))
            if href and "/status/" in href and href not in seen:
                seen.append(href)
        return seen

    def get_main_status_link(self, article) -> str | None:
        candidates = self.get_status_links(article)
        return candidates[0] if candidates else None

    # -----------------------------
    # Tweet utils
    # -----------------------------
    def expand_tweet_if_truncated(self, article):
        try:
            show_more = article.find_element(
                By.CSS_SELECTOR, '[data-testid="tweet-text-show-more-link"]'
            )
            self.driver.execute_script("arguments[0].click();", show_more)
            human_sleep(0.8, 1.8)
        except (NoSuchElementException, StaleElementReferenceException):
            pass

    @staticmethod
    def extract_tweet_text(article) -> str:
        try:
            tweet_body = article.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            return tweet_body.text.strip()
        except NoSuchElementException:
            return article.text.strip()
        except StaleElementReferenceException:
            return article.text.strip()

    @staticmethod
    def detect_tweet_type(article_text: str) -> str:
        header = "\n".join(article_text.split("\n")[:4]).lower()
        if "reposteó" in header or "reposted" in header:
            return "repost"
        if "en respuesta a" in header or "replying to" in header or "respondiendo a" in header:
            return "reply"
        return "original"

    def is_pinned_tweet(self, article) -> bool:
        try:
            social_context = article.find_element(
                By.CSS_SELECTOR, '[data-testid="socialContext"]'
            )
            text = social_context.text.lower()
            return any(word in text for word in ("pinned", "fijado", "destacado"))
        except (NoSuchElementException, StaleElementReferenceException):
            return False

    @staticmethod
    def parse_tweet_datetime(tweet_time: str) -> datetime | None:
        if not tweet_time:
            return None
        try:
            return datetime.fromisoformat(tweet_time.replace("Z", "+00:00"))
        except ValueError:
            return None

    # -----------------------------
    # Referenced context
    # -----------------------------
    def get_referenced_context(self, tweet_url: str, scroll_position: int) -> dict:
        context = {
            "original_tweet_text": None,
            "original_tweet_url": None,
            "original_author": None,
        }
        previous_url = self.driver.current_url

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.driver.get(tweet_url)
                self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                human_sleep(self.config.scroll_pause_min, self.config.scroll_pause_max)

                articles = self.driver.find_elements(By.TAG_NAME, "article")
                target_index = None
                for i, article in enumerate(articles):
                    try:
                        candidate_url = self.get_main_status_link(article)
                        if candidate_url == self.normalize_url(tweet_url):
                            target_index = i
                            break
                    except StaleElementReferenceException:
                        continue

                if target_index is not None and target_index > 0:
                    original_article = articles[target_index - 1]
                    original_text = self.extract_tweet_text(original_article)
                    original_url = self.get_main_status_link(original_article)
                    original_author = None
                    try:
                        author_elem = original_article.find_element(
                            By.CSS_SELECTOR, '[data-testid="User-Name"]'
                        )
                        original_author = author_elem.text.split("\n")[0].strip()
                    except NoSuchElementException:
                        lines = [l.strip() for l in original_text.split("\n") if l.strip()]
                        original_author = lines[0] if lines else None

                    context["original_tweet_text"] = original_text
                    context["original_tweet_url"] = original_url
                    context["original_author"] = original_author
                    break

            except TimeoutException:
                log.warning(f"Timeout getting context for {tweet_url} (attempt {attempt})")
                if attempt == self.config.max_retries:
                    log.error(f"Giving up on context for: {tweet_url}")
            except Exception as e:
                log.error(f"Unexpected error getting context from {tweet_url}: {e}")
                break

        self.driver.get(previous_url)
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
            self.driver.execute_script("window.scrollTo(0, arguments[0]);", scroll_position)
        except TimeoutException:
            log.warning("Could not restore previous search page after context fetch.")

        human_sleep(1.5, 3.0)
        return context

    # -----------------------------
    # Scrape one search window
    # -----------------------------
    def scrape_window(
        self,
        username: str,
        since: datetime,
        until: datetime,
        seen_links: set,
        tweets_data: list,
    ) -> int:
        """
        Scrapea una ventana de búsqueda (since → until) para un usuario.
        Devuelve el número de tweets nuevos encontrados en esta ventana.
        """
        cfg = self.config
        url = build_search_url(username, since, until)
        log.info(f"[{username}] Ventana: {since.date()} → {until.date()} | URL: {url}")

        # Cargar la página de búsqueda
        loaded = False
        for attempt in range(1, cfg.max_retries + 1):
            try:
                self.driver.get(url)
                # Esperamos el primer artículo O el mensaje de "no results"
                self.wait.until(
                    lambda d: d.find_elements(By.TAG_NAME, "article")
                    or d.find_elements(By.CSS_SELECTOR, '[data-testid="empty_state_header_text"]')
                )
                loaded = True
                break
            except TimeoutException:
                log.warning(
                    f"[{username}] Timeout cargando ventana "
                    f"{since.date()}→{until.date()} (intento {attempt}/{cfg.max_retries})"
                )
                human_sleep(10, 20)

        if not loaded:
            log.error(f"[{username}] No se pudo cargar la ventana. Saltando.")
            return 0

        # Comprobar si la búsqueda está vacía
        empty = self.driver.find_elements(
            By.CSS_SELECTOR, '[data-testid="empty_state_header_text"]'
        )
        if empty:
            log.info(f"[{username}] Ventana vacía {since.date()}→{until.date()}, siguiente.")
            return 0

        human_sleep(2.5, 5.0)

        rounds_without_new_ids = 0
        window_new_tweets = 0

        while True:
            round_new_ids = 0
            articles = self.driver.find_elements(By.TAG_NAME, "article")
            log.info(
                f"[{username}] [{since.date()}→{until.date()}] "
                f"Articles on screen: {len(articles)}"
            )

            for idx in range(len(articles)):
                try:
                    articles = self.driver.find_elements(By.TAG_NAME, "article")
                    if idx >= len(articles):
                        continue

                    article = articles[idx]
                    tweet_link = self.get_main_status_link(article)

                    if not tweet_link or tweet_link in seen_links:
                        continue

                    try:
                        time_elem = article.find_element(By.TAG_NAME, "time")
                        tweet_time = time_elem.get_attribute("datetime")
                    except NoSuchElementException:
                        tweet_time = None

                    tweet_dt = self.parse_tweet_datetime(tweet_time)

                    seen_links.add(tweet_link)
                    round_new_ids += 1

                    # Filtro por fecha (por si X devuelve resultados ligeramente fuera del rango)
                    if tweet_dt and tweet_dt > cfg.date_to:
                        log.info(
                            f"[{username}] Tweet {tweet_dt.date()} posterior al rango, descartando."
                        )
                        continue

                    if tweet_dt and tweet_dt < cfg.date_from:
                        if self.is_pinned_tweet(article):
                            log.info(f"[{username}] Tweet fijado antiguo, ignorando.")
                        else:
                            log.info(
                                f"[{username}] Tweet {tweet_dt.date()} anterior al rango, descartando."
                            )
                        continue

                    self.expand_tweet_if_truncated(article)
                    text = self.extract_tweet_text(article)
                    tweet_id = tweet_link.rstrip("/").split("/")[-1]
                    tweet_type = self.detect_tweet_type(article.text)
                    is_reply = tweet_type == "reply"
                    is_repost = tweet_type == "repost"

                    referenced_context = None
                    if cfg.fetch_context_for_replies and (is_reply or is_repost):
                        log.info(f"[{username}] Fetching context: {tweet_link}")
                        scroll_y = self.driver.execute_script("return window.scrollY;")
                        human_sleep(2.0, 4.5)
                        referenced_context = self.get_referenced_context(
                            tweet_link, scroll_position=scroll_y
                        )

                    tweet_record = {
                        "public_figure": username,
                        "tweet_id": tweet_id,
                        "tweet_type": tweet_type,
                        "is_reply": is_reply,
                        "is_repost": is_repost,
                        "tweet_text": text,
                        "datetime": tweet_time,
                        "tweet_url": tweet_link,
                        "referenced_tweet_context": referenced_context,
                        "scrape_date": datetime.now(timezone.utc).isoformat(),
                        # Extra: ventana de búsqueda que lo encontró
                        "search_window": f"{since.date()}_{until.date()}",
                    }

                    tweets_data.append(tweet_record)
                    window_new_tweets += 1

                    log.info(
                        f"[{username}] Total acumulado: {len(tweets_data)} tweets | "
                        f"último: {tweet_dt.date() if tweet_dt else 'unknown'}"
                    )

                    if len(tweets_data) % cfg.save_every_n_tweets == 0:
                        self._save_progress(username, tweets_data)

                except StaleElementReferenceException:
                    continue
                except Exception as e:
                    log.error(f"Error procesando article para {username}: {e}")
                    continue

            # Control de parada por estancamiento dentro de la ventana
            if round_new_ids == 0:
                rounds_without_new_ids += 1
            else:
                rounds_without_new_ids = 0

            if rounds_without_new_ids >= cfg.max_rounds_without_new_ids:
                log.info(
                    f"[{username}] Ventana {since.date()}→{until.date()} agotada "
                    f"({rounds_without_new_ids} rondas sin nuevos IDs)."
                )
                break

            # Scroll para cargar más resultados en esta ventana
            moved = self.scroll_to_last_article()
            if not moved:
                self.human_scroll()

            human_sleep(cfg.scroll_pause_min, cfg.scroll_pause_max)

        return window_new_tweets

    # -----------------------------
    # Scrape a single user (todas las ventanas)
    # -----------------------------
    def scrape_user(self, username: str) -> list[dict]:
        cfg = self.config

        # Generar todas las ventanas temporales
        windows = generate_date_windows(cfg.date_from, cfg.date_to, cfg.search_window_days)
        log.info(
            f"[{username}] Rango: {cfg.date_from.date()} → {cfg.date_to.date()} | "
            f"{len(windows)} ventanas de {cfg.search_window_days} días."
        )

        tweets_data: list[dict] = []
        seen_links: set[str] = set()

        for i, (since, until) in enumerate(windows):
            log.info(
                f"[{username}] === Ventana {i + 1}/{len(windows)}: "
                f"{since.date()} → {until.date()} ==="
            )

            new_in_window = self.scrape_window(
                username, since, until, seen_links, tweets_data
            )
            log.info(
                f"[{username}] Ventana {i + 1} terminada. "
                f"Tweets nuevos: {new_in_window} | Total: {len(tweets_data)}"
            )

            # Guardado al final de cada ventana
            self._save_progress(username, tweets_data)

            # Pausa entre ventanas (excepto la última)
            if i < len(windows) - 1:
                pause = random.uniform(
                    cfg.pause_between_windows_min,
                    cfg.pause_between_windows_max,
                )
                log.info(f"[{username}] Pausa entre ventanas: {pause:.1f}s")
                time.sleep(pause)

        log.info(
            f"[{username}] Scraping finalizado. "
            f"Total tweets en rango: {len(tweets_data)}"
        )
        return tweets_data

    # -----------------------------
    # Guardado incremental
    # -----------------------------
    def _save_progress(self, username: str, tweets: list[dict]):
        path = os.path.join(self.config.output_folder, f"{username}_tweets.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tweets, f, ensure_ascii=False, indent=2)
        except IOError as e:
            log.error(f"Could not save progress for {username}: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    scraper = XSearchScraper(CONFIG)
    all_tweets = []

    try:
        for i, username in enumerate(CONFIG.usernames):
            tweets = scraper.scrape_user(username)
            all_tweets.extend(tweets)
            log.info(f"Finished {username}: {len(tweets)} tweets collected.")

            if i < len(CONFIG.usernames) - 1:
                wait = random.uniform(
                    CONFIG.pause_between_users_min,
                    CONFIG.pause_between_users_max,
                )
                log.info(f"Esperando {wait:.0f}s antes del siguiente perfil...")
                time.sleep(wait)

    finally:
        scraper.quit()

    combined_file = os.path.join(CONFIG.output_folder, "all_public_figures_tweets.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, ensure_ascii=False, indent=2)

    log.info(f"Total tweets collected: {len(all_tweets)}")
    log.info(f"Combined file saved: {combined_file}")


if __name__ == "__main__":
    main()