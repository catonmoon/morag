# Сравнение языков программирования

Ниже приведена подробная сравнительная таблица языков программирования по множеству критериев: производительность, экосистема, типизация, области применения и особенности.

## Полная сравнительная таблица

| Язык | Год создания | Автор/Компания | Парадигма | Типизация | Компиляция | Сборка мусора | Производительность (отн.) | Основные области | Пакетный менеджер | Популярность (TIOBE 2024) | Уровень входа | Конкурентность | Системы типов | Null Safety | Макросы | Метапрограммирование | Зрелость экосистемы |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python | 1991 | Guido van Rossum | Мультипарадигменный: ООП, функциональный, процедурный | Динамическая, строгая | Интерпретируемый (CPython), JIT (PyPy) | Да (reference counting + GC) | 1x (базовая) | ML/AI, веб, скрипты, автоматизация, data science | pip, poetry, conda | #1 | Низкий | asyncio, threading (GIL), multiprocessing | Duck typing, type hints (PEP 484) | Нет (None) | Нет | Декораторы, метаклассы, __init_subclass__ | Очень зрелая |
| JavaScript | 1995 | Brendan Eich (Netscape) | Мультипарадигменный: ООП (прототипное), функциональный | Динамическая, слабая | JIT (V8, SpiderMonkey, JavaScriptCore) | Да (mark-and-sweep) | 3-5x | Веб (фронтенд и бэкенд), мобильные (React Native), десктоп (Electron) | npm, yarn, pnpm | #6 | Низкий-средний | Event loop, Web Workers, Worker Threads | Duck typing, JSDoc | Нет (null + undefined) | Нет | Proxy, Reflect, Symbol | Очень зрелая |
| TypeScript | 2012 | Microsoft (Anders Hejlsberg) | Мультипарадигменный: ООП, функциональный | Статическая (structural), опциональная | Транспилируется в JS | Наследует от JS | Как JS | Веб-приложения, Node.js backend, инструменты разработчика | npm (тот же что JS) | #10 | Средний | Наследует от JS | Structural typing, generics, union/intersection types, conditional types | strict null checks (опционально) | Нет | Декораторы, conditional types, mapped types | Зрелая |
| Java | 1995 | James Gosling (Sun Microsystems) | ООП, функциональный (с Java 8) | Статическая, строгая | Компилируемый (JVM bytecode + JIT) | Да (G1, ZGC, Shenandoah) | 10-15x | Enterprise, Android, микросервисы, Big Data | Maven, Gradle | #4 | Средний | Threads, virtual threads (Project Loom), ForkJoinPool | Nominal typing, generics (type erasure) | Optional (Java 8), null по умолчанию | Нет | Reflection, annotation processing | Очень зрелая |
| C# | 2000 | Microsoft (Anders Hejlsberg) | Мультипарадигменный: ООП, функциональный | Статическая, строгая | Компилируемый (CLR/IL + JIT) | Да (generational GC) | 10-15x | Enterprise, игры (Unity), десктоп (WPF/MAUI), веб (ASP.NET) | NuGet | #5 | Средний | async/await, Task, Parallel.ForEach, channels | Nominal typing, generics (reified) | Nullable reference types (C# 8+) | Нет | Source generators, reflection, expressions | Очень зрелая |
| Go | 2009 | Google (Rob Pike, Ken Thompson, Robert Griesemer) | Процедурный, конкурентный | Статическая, строгая | Компилируемый (нативный) | Да (concurrent tri-color mark-and-sweep) | 15-20x | Облачная инфраструктура, микросервисы, CLI, DevOps | go mod | #8 | Низкий-средний | Goroutines + channels (CSP) | Structural typing (interfaces), без generics до 1.18 | Нет (nil) | Нет | go generate, code generation | Зрелая |
| Rust | 2010 | Mozilla (Graydon Hoare) | Мультипарадигменный: функциональный, императивный | Статическая, строгая | Компилируемый (LLVM, нативный) | Нет (ownership + borrowing) | 20-30x | Системное программирование, WebAssembly, CLI, embedded, сетевые сервисы | Cargo | #14 | Высокий | async/await (tokio, async-std), threads, Send/Sync traits | Nominal typing, generics (monomorphization), trait system | Option<T> вместо null | Да (macro_rules!, proc macros) | Макросы, const generics, trait impls | Растущая |
| C++ | 1979 | Bjarne Stroustrup | Мультипарадигменный: ООП, обобщённый, процедурный | Статическая, строгая | Компилируемый (нативный) | Нет (ручное управление, RAII, smart pointers) | 25-35x | Игры, ОС, драйверы, HPC, embedded, финтех | vcpkg, conan | #2 | Высокий | std::thread, std::async, coroutines (C++20) | Nominal typing, templates (Turing-complete) | Нет (nullptr) | Да (препроцессор + templates) | Templates, SFINAE, concepts (C++20) | Очень зрелая |
| C | 1972 | Dennis Ritchie (Bell Labs) | Процедурный | Статическая, слабая | Компилируемый (нативный) | Нет (ручное управление) | 30-40x | ОС, ядра, драйверы, embedded, протоколы | Нет стандартного (make, cmake) | #3 | Высокий | POSIX threads, select/poll/epoll | Minimal (struct, typedef) | Нет (NULL) | Да (препроцессор) | Препроцессор, _Generic (C11) | Очень зрелая |
| Kotlin | 2011 | JetBrains | Мультипарадигменный: ООП, функциональный | Статическая, строгая | JVM bytecode, JS, Native (LLVM) | Да (JVM GC / Native: нет) | 10-15x (JVM) | Android (official), backend (Ktor, Spring), мультиплатформенный | Gradle, Maven | #15 | Средний | Coroutines (structured concurrency), Flow | Nominal typing, generics (reified inline), sealed classes | Null safety встроен (? / !!) | Нет | Compiler plugins, KSP | Зрелая |
| Swift | 2014 | Apple (Chris Lattner) | Мультипарадигменный: ООП, функциональный, протокол-ориентированный | Статическая, строгая | Компилируемый (LLVM, нативный) | ARC (automatic reference counting) | 15-25x | iOS/macOS разработка, серверный Swift (Vapor), системное | Swift Package Manager | #16 | Средний | async/await, actors, structured concurrency (Swift 5.5+) | Nominal typing, generics, protocol-oriented | Optional<T> (built-in) | Нет | Property wrappers, result builders | Зрелая (Apple ecosystem) |
| Scala | 2004 | Martin Odersky (EPFL) | Мультипарадигменный: ООП + функциональный | Статическая, строгая | JVM bytecode (Scala.js, Scala Native) | Да (JVM GC) | 10-15x | Big Data (Spark), backend, DSL, функциональное программирование | sbt, Mill | ~#30 | Высокий | Akka actors, Cats Effect, ZIO (fiber-based) | Nominal + structural (Scala 3), generics, path-dependent types, HKT | Option[T] | Да (Scala 3: inline, macros) | Compile-time macros, implicits, given/using | Зрелая |
| Ruby | 1995 | Yukihiro Matsumoto | Мультипарадигменный: ООП (всё объект), функциональный | Динамическая, строгая | Интерпретируемый (CRuby), JIT (YJIT) | Да (mark-and-sweep) | 1-2x | Веб (Rails), скрипты, автоматизация, DevOps | Bundler, RubyGems | #18 | Низкий | Threads (с GVL), Ractors (Ruby 3), Fibers | Duck typing, RBS type signatures | Нет (nil) | Нет | method_missing, define_method, open classes | Зрелая |
| PHP | 1995 | Rasmus Lerdorf | Мультипарадигменный: ООП, процедурный | Динамическая, слабая (строже с 8.0) | Интерпретируемый (Zend), JIT (PHP 8.0) | Да (reference counting + cycle collector) | 2-4x | Веб-бэкенд (WordPress, Laravel, Symfony), CMS | Composer | #7 | Низкий | Fibers (PHP 8.1), pcntl_fork, pthreads | Gradual typing (PHP 8.0+), generics (PHPStan) | Null, ?Type (PHP 7.1), union types (8.0) | Нет | Нет | Reflection, attributes (PHP 8.0) | Очень зрелая |
| Elixir | 2011 | José Valim | Функциональный, конкурентный | Динамическая, строгая | Компилируемый (BEAM VM bytecode) | Да (per-process GC) | 2-5x | Веб (Phoenix), телеком, real-time системы, IoT | Hex, Mix | ~#40 | Средний-высокий | Процессы BEAM (lightweight, миллионы), OTP supervisors, GenServer | Duck typing, @spec typespecs, Dialyzer | Нет (nil) | Да (мощная макро-система) | Макросы (AST transformation), protocols, behaviours | Растущая |
| Haskell | 1990 | Комитет (Simon Peyton Jones и др.) | Чисто функциональный | Статическая, строгая | Компилируемый (GHC, нативный) | Да (generational) | 10-20x | Академия, финтех, компиляторы, формальная верификация | Cabal, Stack | ~#35 | Очень высокий | STM, async, par, green threads (GHC RTS) | Hindley-Milner + extensions, GADTs, type families, HKT | Maybe a | Нет | Template Haskell, GHC plugins | Зрелая (нишевая) |
| Zig | 2015 | Andrew Kelley | Процедурный, системный | Статическая, строгая | Компилируемый (LLVM, нативный) | Нет (ручное управление, arena allocators) | 25-35x | Системное, embedded, замена C, игры | Встроенный (zig build) | ~#50 | Высокий | async (stackless coroutines), threads | Structural (comptime), generics через comptime | Optional (?T), error union (E!T) | Нет | comptime (compile-time code execution) | Ранняя |
| Nim | 2008 | Andreas Rumpf | Мультипарадигменный: процедурный, ООП, функциональный | Статическая, строгая | Компилируемый (через C/C++/JS) | Опционально (ARC/ORC/none) | 20-30x | Системное, скрипты, веб, игры, CLI | Nimble | ~#60 | Средний | async/await, threads, channels | Structural + nominal, generics | Option[T] (библиотечная) | Да (мощная AST-макросистема) | Templates, macros, compile-time evaluation | Ранняя |
| Dart | 2011 | Google (Lars Bak, Kasper Lund) | Мультипарадигменный: ООП, функциональный | Статическая, строгая (sound null safety) | JIT (VM) + AOT (нативный) | Да (generational) | 5-10x | Мобильные (Flutter), веб, серверный | pub | ~#25 | Низкий-средний | Isolates, async/await, streams | Nominal typing, generics (reified) | Sound null safety (Dart 2.12+) | Нет | Нет | Source generation (build_runner), mirrors (ограниченно) | Зрелая (Flutter) |
| Julia | 2012 | Jeff Bezanson, Stefan Karpinski и др. (MIT) | Мультипарадигменный: multiple dispatch, функциональный | Динамическая (с опциональными аннотациями), строгая | JIT (LLVM) | Да (generational, incremental) | 15-25x (числ. вычисления) | Научные вычисления, ML, data science, HPC | Pkg (встроенный) | ~#30 | Средний | Tasks (green threads), distributed computing, @threads | Multiple dispatch, parametric types | Nothing (subtype of all) | Да (@macro) | Макросы (AST), generated functions, metaprogramming | Растущая |
| Lua | 1993 | PUC-Rio (Roberto Ierusalimschy и др.) | Мультипарадигменный: процедурный, ООП (через метатаблицы) | Динамическая, слабая (приведение типов) | Интерпретируемый, JIT (LuaJIT) | Да (incremental mark-and-sweep) | 5-15x (LuaJIT) | Встраиваемый скриптинг (игры, Redis, Nginx), конфигурация | LuaRocks | ~#25 | Низкий | Coroutines (кооперативная), без preemptive multithreading | Duck typing, без type annotations | Нет (nil) | Нет | Нет | Метатаблицы, environments, debug library | Зрелая (нишевая) |

## Детальное сравнение производительности

Ниже приведены результаты бенчмарков для типичных задач: сортировка массива из 10 миллионов элементов, HTTP-сервер (requests/sec), JSON-парсинг и работа с регулярными выражениями.

| Язык | Сортировка 10M int (мс) | HTTP req/sec (hello world) | JSON parse 1MB (мс) | Regex scan 1GB (сек) | Память idle (MB) | Время запуска (мс) | Размер бинарника (hello, MB) | Потребление CPU при idle (%) |
|---|---|---|---|---|---|---|---|---|
| C | 450 | 2,400,000 | 35 | 2.1 | 0.5 | 1 | 0.01 | 0.0 |
| C++ | 460 | 2,200,000 | 38 | 2.3 | 1.2 | 1 | 0.02 | 0.0 |
| Rust | 470 | 2,100,000 | 40 | 2.4 | 1.0 | 1 | 0.3 | 0.0 |
| Zig | 465 | 2,000,000 | 42 | 2.5 | 0.8 | 1 | 0.2 | 0.0 |
| Go | 550 | 1,200,000 | 55 | 8.5 | 3.5 | 5 | 1.8 | 0.1 |
| Java (GraalVM) | 520 | 900,000 | 45 | 3.8 | 45 | 50 | 15 | 0.3 |
| C# (.NET 8) | 530 | 850,000 | 48 | 4.0 | 25 | 30 | 12 | 0.2 |
| Kotlin (JVM) | 525 | 880,000 | 46 | 3.9 | 48 | 55 | N/A (JVM) | 0.3 |
| Swift | 510 | 700,000 | 50 | 3.5 | 5 | 3 | 0.5 | 0.0 |
| Dart (AOT) | 650 | 450,000 | 60 | 5.5 | 8 | 10 | 5 | 0.1 |
| Scala (JVM) | 530 | 800,000 | 47 | 4.0 | 60 | 80 | N/A (JVM) | 0.4 |
| Julia | 490 | 350,000 | 55 | 4.2 | 120 | 300 | N/A (JIT) | 0.5 |
| Nim | 480 | 1,500,000 | 43 | 2.8 | 1.5 | 2 | 0.1 | 0.0 |
| Elixir (BEAM) | 2200 | 250,000 | 180 | 15.0 | 30 | 200 | N/A (VM) | 0.2 |
| Haskell | 600 | 600,000 | 55 | 3.0 | 10 | 5 | 2.0 | 0.1 |
| JavaScript (V8) | 800 | 550,000 | 65 | 6.0 | 15 | 30 | N/A (V8) | 0.3 |
| TypeScript (Bun) | 810 | 500,000 | 68 | 6.2 | 18 | 35 | N/A (VM) | 0.3 |
| Python (CPython) | 8500 | 15,000 | 250 | 12.0 | 8 | 30 | N/A (interp) | 0.1 |
| Python (PyPy) | 1200 | 80,000 | 120 | 8.0 | 50 | 200 | N/A (JIT) | 0.3 |
| Ruby (YJIT) | 3500 | 45,000 | 180 | 10.0 | 12 | 40 | N/A (interp) | 0.1 |
| PHP 8.3 | 2800 | 120,000 | 150 | 9.0 | 6 | 10 | N/A (interp) | 0.1 |
| Lua (LuaJIT) | 900 | 400,000 | 90 | 7.0 | 2 | 2 | N/A (JIT) | 0.0 |

Примечание: все замеры приблизительные и зависят от конкретной реализации, версии компилятора и оборудования. Данные приведены для ориентировочного сравнения на типичном x86_64 сервере (16 ядер, 64 ГБ RAM).

## Матрица совместимости фреймворков

Какие фреймворки доступны для каждого языка в различных областях.

| Язык | Веб-бэкенд | Веб-фронтенд | Мобильные | Десктоп | ML/AI | Базы данных (ORM) | Тестирование | CI/CD | Контейнеризация | Облачные SDK |
|---|---|---|---|---|---|---|---|---|---|---|
| Python | Django, Flask, FastAPI, Starlette, Sanic, Tornado, Bottle, Falcon, Quart, Litestar | Brython, Transcrypt, PyScript, Streamlit, Gradio | Kivy, BeeWare, Flet | PyQt, Tkinter, wxPython, Dear PyGui, Flet | PyTorch, TensorFlow, scikit-learn, JAX, Keras, Hugging Face, spaCy, NLTK, XGBoost | SQLAlchemy, Django ORM, Tortoise, Peewee, Pony | pytest, unittest, nose2, hypothesis, tox | GitHub Actions, GitLab CI, Jenkins, CircleCI | Docker (slim/alpine), Poetry, pip-tools | boto3, google-cloud, azure-sdk |
| JavaScript | Express, Fastify, Koa, Hapi, NestJS, AdonisJS, Hono | React, Vue, Angular, Svelte, Solid, Qwik, Astro, Next.js, Nuxt | React Native, Ionic, NativeScript, Expo | Electron, Tauri, NW.js | TensorFlow.js, ONNX.js, Brain.js, ml5.js | Prisma, TypeORM, Sequelize, Knex, Drizzle, Mongoose | Jest, Vitest, Mocha, Cypress, Playwright, Testing Library | GitHub Actions, CircleCI, Vercel, Netlify | Docker (node:alpine), pnpm | aws-sdk, @google-cloud, @azure |
| Java | Spring Boot, Quarkus, Micronaut, Jakarta EE, Vert.x, Helidon, Play | Vaadin, GWT (устаревший) | Android SDK, Android Jetpack | JavaFX, Swing, SWT | DL4J, Weka, Apache Mahout, Tribuo, ONNX Runtime | Hibernate, JPA, jOOQ, MyBatis, Spring Data | JUnit 5, TestNG, Mockito, AssertJ, Testcontainers | Maven/Gradle plugins, Jenkins, GitHub Actions | Docker (eclipse-temurin), GraalVM native | AWS SDK v2, Google Cloud Java, Azure SDK |
| Go | Gin, Echo, Fiber, Chi, Gorilla Mux, Hertz, net/http | — (WASM через syscall/js) | Fyne (experimental), gomobile | Fyne, Wails, Gio | GoLearn, Gorgonia (ограниченно) | GORM, Ent, sqlx, sqlc, Bun | testing (stdlib), testify, gomock, go-cmp | GitHub Actions, GoReleaser, Dagger | Docker (scratch/distroless/alpine), ko | aws-sdk-go-v2, google-cloud-go, azure-sdk-for-go |
| Rust | Actix-web, Axum, Rocket, Warp, Tide, Poem | Yew, Leptos, Dioxus, Sycamore (WASM) | — (через FFI) | Tauri, Iced, egui, Slint | candle, tch-rs, burn, linfa | Diesel, SQLx, SeaORM, rusqlite | cargo test, proptest, mockall, rstest | GitHub Actions, cargo-release | Docker (scratch/distroless), cross | aws-sdk-rust, google-cloud-rust, azure_sdk |

## Рекомендации по выбору

Выбор языка программирования зависит от множества факторов. Не существует универсально лучшего языка — каждый оптимален в своей нише. При выборе стоит учитывать требования к производительности, доступность разработчиков, зрелость экосистемы и долгосрочную поддержку.
