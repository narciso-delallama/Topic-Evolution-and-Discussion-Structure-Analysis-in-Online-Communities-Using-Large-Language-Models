from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging
 
# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
 
# -----------------------------
# Config — debe coincidir con scraper_mejorado.py
# -----------------------------
CHROME_PROFILE_PATH = r"C:\Users\narci\chrome_selenium_profile"
 
# -----------------------------
# Login
# -----------------------------
def main():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")
 
    log.info("Abriendo Chrome con perfil persistente...")
    driver = webdriver.Chrome(options=options)
 
    try:
        driver.get("https://x.com/login")
        log.info("Navega a X.com e inicia sesión manualmente.")
        input("\n>>> Pulsa ENTER cuando hayas completado el login... ")
        log.info(f"Sesión guardada en: {CHROME_PROFILE_PATH}")
        log.info("Ya puedes cerrar y ejecutar el scraper.")
    except Exception as e:
        log.error(f"Error durante el login: {e}")
    finally:
        driver.quit()
        log.info("Navegador cerrado.")
 
 
if __name__ == "__main__":
    main()