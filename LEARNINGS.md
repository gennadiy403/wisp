# Learnings — govori

Non-obvious находки по проекту. Растёт со временем; самые свежие сверху.

## 2026-04-23 — fn-down/fn-up race condition + Whisper hallucination filter + launchagent debugging

### fn-down/fn-up race: state mutation должен быть в callback, а не в thread
**Контекст:** коммиты 0603c59 + 0775cd2 пытались починить quick-tap race. Пользователь жаловался: "при нажатии fn не всегда реагирует старт записи" — логи показывали много `[mode]` без последующего `Recording...`.
**Находка:** старая архитектура раскидывала state между потоками: fn-down callback спавнил `start_recording` thread, который *асинхронно* ставил `recording=True` под `_state_lock`. Если fn-up приходил раньше, чем thread успевал взять lock (PortAudio init может блокировать сотни мс), ветка `elif recording:` проваливалась в пустоту — микрофон стартовал после fn-up и висел навсегда.
**Как применить:** для async macOS-daemons с CGEventTap — делать **sync state mutation в callback** под lock, а в thread выносить только блокирующий I/O (`sd.InputStream`, API calls). Тогда callback и следующий событийный callback всегда видят актуальный state. См. `govori.py:_start_mic_stream` — сейчас mic init в thread, но `recording=True/False` ставится синхронно в `cg_event_callback`.

### `not transcribing` в guard условии = скрытая регрессия UX
**Контекст:** при переделке race я добавил `if not recording and not transcribing:` в guard для старта новой записи.
**Находка:** `transcribing=True` живёт 5-10 сек пока идёт API-запрос к Whisper. Всё это время новые нажатия fn игнорировались — пользователь видел `[mode]` без `Recording...`. Старое поведение разрешало перекрывать идущую транскрипцию новой записью.
**Как применить:** при синхронизации state не добавляй новых условий в существующие guard — только воспроизводи старую семантику. Race-фиксы должны быть поведенчески-нейтральны. Единственное валидное условие для блокировки fn-down — `recording=True` (уже идёт запись).

### Две копии govori = источник "зависаний" CGEventTap
**Контекст:** пользователь отлаживал race внутри одного процесса, пока я не проверил `ps aux`.
**Находка:** был запущен launchagent (com.user.govori) + терминальная копия (`./govori` из shell). Оба регистрировали `CGEventTap` на fn. macOS доставляет событие обоим; один читает микрофон, второй в то же время видит `recording=False` и инициирует свой, потом оба конфликтуют. Симптом — "зависание", внешне неотличимое от внутрипроцессного race.
**Как применить:** ПЕРВЫЙ шаг при debugging fn/HUD/recording бага — `pgrep -lf 'python.*govori\.py'`. Если больше одного — убить лишнее до любой правки кода.

### `_ensure_singleton` ловит false-positives от pgrep
**Контекст:** после `launchctl kickstart -k ...` процесс не запускался — в логе `! Govori is already running (PID XXXXX). Another instance is active — refusing to start.`. Но `ps` не показывал такого PID.
**Находка:** `_ensure_singleton` использует `pgrep -f govori.py` — паттерн матчит *любую* командную строку с подстрокой `govori.py`, включая короткоживущие pipeline-процессы вроде `ps aux | grep govori.py` или `grep govori.py file`. Они исчезают до `ps -p`, но на момент `pgrep` существуют.
**Как применить:** сузить паттерн до `python.*govori\.py` в `_find_other_govori_pids` (govori.py:2882). Либо проверять по command (не args) через `pgrep -x python` + отдельный фильтр на аргументы через `/proc`-аналог.

### Whisper `language="ru"` не гарантирует русский вывод
**Контекст:** добавили конкретные галлюцинации ("Субтитры создавал DimaTorzok") в фильтр. Пользователь: "только русский, термины на англ OK, никаких других языков".
**Находка:** даже с `language="ru"` и `whisper_prompt` модель периодически выдаёт CJK-иероглифы (`ご視聴ありがとうございました`), арабицу, деванагари на тишине/шуме. Force language работает как hint, не constraint.
**Как применить:** фильтр на уровне вывода через Unicode-blocks. См. `_FOREIGN_SCRIPT_RE` в `govori.py:_is_hallucination` — blacklist CJK/Hangul/Arabic/Hebrew/Indic/Thai/Georgian/Ethiopic, любой символ из этих блоков → весь текст отбрасывается. Кириллица+латиница+цифры+пунктуация — allow-listed неявно.

### `_state_lock` удерживается во время блокирующих PortAudio syscalls
**Контекст:** анализ race — почему `start_recording` задерживается относительно fn-up.
**Находка:** `sd.InputStream.start()` и `audio_stream.stop()/close()` — блокирующие PortAudio syscalls, каждый может занимать 100-500мс на холодном старте. В оригинальном `start_recording` они вызывались *внутри* `with _state_lock:` — всё это время любой другой код, пытающийся взять lock (включая event callback для следующего fn), блокировался.
**Как применить:** для macOS daemons с CGEventTap — никогда не держать shared lock под блокирующим syscall. Event callback должен возвращаться за миллисекунды, иначе system events начнут дропаться или tap отключится. Держи lock только над RAM-операциями; I/O — отдельно, желательно в thread. В govori это решено частично: `_start_mic_stream` всё ещё держит lock под `sd.InputStream.start()`, но это терпимо т.к. callback больше не ждёт этот lock (state уже установлен).
