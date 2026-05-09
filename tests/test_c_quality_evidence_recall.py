from __future__ import annotations

from app.application.use_cases.build_coverage_units import CoverageUnitBuilder
from app.application.use_cases.adaptive_candidate_selector import select_candidates
from app.application.use_cases.retrieve_candidates import CandidateRetriever
from app.core.config import CoverageRetrievalConfig
from app.domain.c_quality_enums import CoverageUnitType, RequirementType
from app.domain.c_quality_models import CoverageUnit, RequirementUnit, RetrievedCandidate
from app.infrastructure.embeddings.simple import BagOfWordsEmbeddingBackend


def _frag(section_id: str, fragment_id: str, text: str) -> dict:
    return {
        "section_id": section_id,
        "fragment_id": fragment_id,
        "kind": "paragraph",
        "text": text,
    }


def test_coverage_unit_builder_adds_section_window_units():
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [
            {
                "section_id": "2.4",
                "title": "Server interaction architecture",
            }
        ],
        "fragments": [
            _frag("2.4", "f1", "The client exchanges data with the server through REST API."),
            _frag("2.4", "f2", "Request and response payloads use JSON."),
            _frag("2.4", "f3", "Files are uploaded through typed web forms."),
            _frag("2.4", "f4", "The maximum upload size is controlled by backend settings."),
        ],
    }

    units = CoverageUnitBuilder().build(artifact)
    window_units = [u for u in units if u.unit_type == CoverageUnitType.SECTION_WINDOW]

    assert window_units, "builder should add section-window evidence units"
    assert any("REST API" in u.text and "JSON" in u.text for u in window_units)
    # Polyakov-regression: the section title is no longer prepended to
    # the window unit text — it duplicated topic on the wire and
    # poisoned LLM judging on Polyakov-class packages. Title now lives
    # in metadata.section_title, where retrieve_candidates already uses
    # it for the section-prior boost.
    assert any(
        "Server interaction architecture" == (u.metadata or {}).get("section_title")
        for u in window_units
    )


def test_coverage_unit_builder_keeps_cumulative_tail_with_requirement_signals():
    common_prefix = "Architecture components. " + ("same context " * 45)
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [{"section_id": "2.2", "title": "Technology rationale"}],
        "fragments": [
            _frag(
                "2.2",
                "f1",
                common_prefix
                + "The component folder contains files with .ts, .html, .css names.",
            ),
            _frag(
                "2.2",
                "f2",
                common_prefix
                + "TypeScript is the main implementation language, Angular is the"
                " main framework, and source code is stored on GitHub.",
            ),
        ],
    }

    units = CoverageUnitBuilder().build(artifact)
    kept_texts = [u.text for u in units]

    assert any("Angular" in text and "GitHub" in text for text in kept_texts)


def test_exact_aspect_boost_lifts_polyakov_style_technology_evidence():
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
    req = RequirementUnit(
        req_id="r-tech",
        source_document_id="doc-tz",
        text="Source code must be written in TypeScript using Angular.",
        normalized_text="source code must be written in typescript using angular",
        requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
    )
    correct = CoverageUnit(
        unit_id="u-correct",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="TypeScript is the main language. Angular is the main framework.",
        normalized_text="typescript is the main language angular is the main framework",
    )
    distractor = CoverageUnit(
        unit_id="u-distractor",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="The repository UI is modern and supports project browsing.",
        normalized_text="the repository ui is modern and supports project browsing",
    )

    results = retriever.retrieve(req, [distractor, correct])

    assert results[0].unit_id == "u-correct"
    assert results[0].retrieval_score >= 0.25


def test_paragraph_unit_gets_section_title_in_metadata():
    """CoverageUnitBuilder must populate section_title in paragraph unit
    metadata so the retrieval-stage non-impl penalty can use it."""
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [
            {"section_id": "3.1", "title": "Существующие аналоги"},
            {"section_id": "3.2", "title": "Архитектура системы"},
        ],
        "fragments": [
            _frag("3.1", "f1", "Repo.hse.ru является текущим решением ВШЭ."),
            _frag("3.2", "f2", "Система реализована на TypeScript и Angular."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    paragraph_units = [u for u in units if u.unit_type == CoverageUnitType.PARAGRAPH]

    titles_by_text = {u.text: u.metadata.get("section_title") for u in paragraph_units}
    assert any(
        t == "Существующие аналоги"
        for text, t in titles_by_text.items()
        if "Repo.hse.ru" in text
    ), "paragraph from section 3.1 should carry section_title='Существующие аналоги'"
    assert any(
        t == "Архитектура системы"
        for text, t in titles_by_text.items()
        if "TypeScript" in text
    ), "paragraph from section 3.2 should carry section_title='Архитектура системы'"


def test_non_impl_section_penalty_demotes_competitor_analysis_evidence():
    """Evidence from 'Существующие аналоги' must rank below implementation
    evidence even when BoW score is similar."""
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
    req = RequirementUnit(
        req_id="r-iface",
        source_document_id="doc-tz",
        text="Клиентская часть должна представлять пользовательский интерфейс.",
        normalized_text="клиентская часть должна представлять пользовательский интерфейс",
        requirement_type=RequirementType.FUNCTIONAL,
    )
    competitor = CoverageUnit(
        unit_id="u-competitor",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Пользовательский интерфейс сервиса реализован в фирменном стиле ВШЭ.",
        normalized_text="пользовательский интерфейс сервиса реализован в фирменном стиле вшэ",
        metadata={"section_title": "Существующие аналоги"},
    )
    impl = CoverageUnit(
        unit_id="u-impl",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Клиентская часть обеспечивает интерфейс для загрузки файлов.",
        normalized_text="клиентская часть обеспечивает интерфейс для загрузки файлов",
        metadata={"section_title": "Реализация"},
    )

    results = retriever.retrieve(req, [competitor, impl])
    ranked_ids = [r.unit_id for r in results]

    assert ranked_ids[0] == "u-impl", (
        "implementation unit must outrank competitor-analysis unit after penalty"
    )


def test_non_impl_section_penalty_not_applied_to_normal_sections():
    """Penalty must NOT be applied to standard implementation sections."""
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
    req = RequirementUnit(
        req_id="r-api",
        source_document_id="doc-tz",
        text="Входные данные отправляются через REST API в формате JSON.",
        normalized_text="входные данные отправляются через rest api в формате json",
        requirement_type=RequirementType.DATA_IO,
    )
    impl = CoverageUnit(
        unit_id="u-impl",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Данные передаются клиентом через REST API в формате JSON.",
        normalized_text="данные передаются клиентом через rest api в формате json",
        metadata={"section_title": "Архитектура взаимодействия"},
    )
    results = retriever.retrieve(req, [impl])
    assert results, "implementation unit should not be filtered out"
    assert results[0].unit_id == "u-impl"
    assert results[0].retrieval_score > 0.0


def test_section_window_lets_retriever_find_rest_json_across_paragraphs():
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [{"section_id": "2.4", "title": "Architecture"}],
        "fragments": [
            _frag("2.4", "f1", "The client part communicates with the backend through REST API."),
            _frag("2.4", "f2", "All server responses are serialized as JSON."),
            _frag("2.4", "f3", "The user interface is implemented as an Angular SPA."),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
    req = RequirementUnit(
        req_id="r-data",
        source_document_id="doc-tz",
        text="Input data must be sent through REST API in JSON format.",
        normalized_text="input data must be sent through rest api in json format",
        requirement_type=RequirementType.DATA_IO,
    )

    results = retriever.retrieve(req, units)
    top_texts = [u.text for u in units if u.unit_id in {r.unit_id for r in results[:3]}]

    assert any("REST API" in text and "JSON" in text for text in top_texts)


def test_polyakov_pz_topic_anchors_beat_admin_distractors():
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [
            {"section_id": "2.1", "title": "Пользовательские сценарии"},
            {"section_id": "2.2", "title": "Обоснование средств разработки"},
            {"section_id": "2.3", "title": "Архитектура взаимодействия с сервером"},
            {"section_id": "2.4", "title": "Прототипирование интерфейса"},
            {"section_id": "3.3", "title": "Существующие аналоги"},
        ],
        "fragments": [
            _frag(
                "3.3",
                "d1",
                "Администратор имеет возможность выдавать роли другим пользователям, создавать новые коллекции и редактировать их иерархию.",
            ),
            _frag(
                "3.3",
                "d2",
                "Внутрь этой коллекции добавляются все недостающие исследования, а ранее одинокое исследование связывается с новой коллекцией.",
            ),
            _frag(
                "2.1",
                "f1",
                "Пользовательские сценарии включают регистрацию нового пользователя, авторизацию по логину и паролю и работу с личным кабинетом автора.",
            ),
            _frag(
                "2.2",
                "f2",
                "TypeScript используется как основной язык разработки клиентской части, Angular выбран как основной фреймворк, исходный код хранится на GitHub.",
            ),
            _frag(
                "2.3",
                "f3",
                "Взаимодействие клиентской части с сервером реализовано через REST API.",
            ),
            _frag(
                "2.3",
                "f4",
                "Данные запросов и ответов передаются в формате JSON.",
            ),
            _frag(
                "2.4",
                "f5",
                "Макеты интерфейса и прототипы экранов были разработаны в Figma.",
            ),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())

    checks = [
        (
            RequirementUnit(
                req_id="r-auth",
                source_document_id="doc-tz",
                text="Система должна предоставить пользователю следующий набор функций: Регистрация, авторизация и аутентификация.",
                normalized_text="система должна предоставить пользователю следующий набор функций регистрация авторизация и аутентификация",
                requirement_type=RequirementType.SECURITY,
            ),
            "Пользовательские сценарии",
        ),
        (
            RequirementUnit(
                req_id="r-tech",
                source_document_id="doc-tz",
                text="Исходные коды программы должны быть написаны на языке программирования TypeScript с использованием библиотеки Angular.",
                normalized_text="исходные коды программы должны быть написаны на языке программирования typescript с использованием библиотеки angular",
                requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
            ),
            "Обоснование средств разработки",
        ),
        (
            RequirementUnit(
                req_id="r-api",
                source_document_id="doc-tz",
                text="Все входные данные отправляются через REST API в формате JSON.",
                normalized_text="все входные данные отправляются через rest api в формате json",
                requirement_type=RequirementType.DATA_IO,
            ),
            "Архитектура взаимодействия с сервером",
        ),
        (
            RequirementUnit(
                req_id="r-figma",
                source_document_id="doc-tz",
                text="Макет интерфейса должен быть разработан в Figma.",
                normalized_text="макет интерфейса должен быть разработан в figma",
                requirement_type=RequirementType.INTERFACE,
            ),
            "Прототипирование интерфейса",
        ),
    ]

    by_id = {u.unit_id: u for u in units}
    for req, expected_title in checks:
        results = retriever.retrieve(req, units)
        top_units = [by_id[r.unit_id] for r in results[:3]]
        top_titles = {u.metadata.get("section_title") for u in top_units}
        assert expected_title in top_titles
        assert "Существующие аналоги" not in {u.metadata.get("section_title") for u in top_units[:1]}


def test_strong_section_window_can_reach_selector():
    cfg = CoverageRetrievalConfig(
        evidence_strength_strong_threshold=0.45,
        selector_strong_margin=0.08,
        selector_max_k=3,
    )
    req = RequirementUnit(
        req_id="r-api",
        source_document_id="doc-tz",
        text="Все входные данные отправляются через REST API в формате JSON.",
        normalized_text="все входные данные отправляются через rest api в формате json",
        requirement_type=RequirementType.DATA_IO,
    )
    candidates = [
        # Weak-ish atomic distractor.
        # It should not block a much better section window that combines REST + JSON.
        RetrievedCandidate(
            req_id="r-api",
            unit_id="u-atomic",
            target_document_id="doc-pz",
            retrieval_score=0.40,
            unit_type=CoverageUnitType.PARAGRAPH,
        ),
        RetrievedCandidate(
            req_id="r-api",
            unit_id="u-window",
            target_document_id="doc-pz",
            retrieval_score=0.62,
            unit_type=CoverageUnitType.SECTION_WINDOW,
        ),
    ]

    selected = select_candidates(req, candidates, cfg)

    assert any(c.unit_id == "u-window" for c in selected.selected)


def test_polyakov_prepare_shape_ranks_real_pz_evidence_over_report_noise():
    artifact = {
        "document_id": "doc-pz",
        "doc_role": "pz",
        "sections": [
            {"section_id": "4.3", "title": "Сценарии роли зарегистрированный пользователь"},
            {"section_id": "4.9", "title": "Обоснование средств разработки"},
            {"section_id": "4.12", "title": "Архитектура компонентов"},
            {"section_id": "4.13", "title": "Архитектура взаимодействия с сервером"},
            {"section_id": "4.16", "title": "Прототипирование интерфейса"},
            {"section_id": "4.7", "title": "Сценарии роли администратор"},
            {"section_id": "5.1", "title": "Стилизация интерфейса"},
        ],
        "fragments": [
            _frag(
                "4.7",
                "4.7::sent1",
                "Администратор – имеет возможность выдавать роли другим пользователям, "
                "создавать новые коллекции и редактировать их иерархию, удалять уже "
                "принятые исследования или добавлять новые, без необходимости проходить "
                "модерацию, а также просматривать контент в закрытом доступе.",
            ),
            _frag(
                "4.12",
                "4.12::sent3",
                "Среди них: .ts, .html, .css, .scss (последний в случае необходимости), "
                "каждый из которых имеет одинаковое название и расположен в директории "
                "с таким же названием для обеспечения простоты поиска файла и понимания "
                "к какому компоненту он относится.",
            ),
            _frag(
                "5.1",
                "5.1::sent7",
                "После выполнения всех модификаций визуальной и структурной части проекта "
                "было необходимо изменить соответствующие параметры в конфигурационном "
                "файле проекта, в целях применения темы проекта, как основной.",
            ),
            _frag(
                "4.3",
                "4.3::sent1",
                "Авторизация Пользователь вводит в форму свои почту и пароль введённые "
                "при регистрации Нажимает на кнопку входа.",
            ),
            _frag(
                "4.9",
                "4.9::sent2",
                "Языки программирования, языки разметки и фреймворки: TypeScript – как "
                "основной язык программирования Angular – как основной фреймворк "
                "GitHub – как сервис для облачного сохранения исходного кода.",
            ),
            _frag(
                "4.13",
                "4.13::sent1",
                "Взаимодействие клиентской части с серверной осуществляется посредством "
                "REST API с использованием формата данных JSON.",
            ),
            _frag(
                "4.16",
                "4.16::sent6",
                "Составление макетов осуществлялось в Figma.",
            ),
        ],
    }
    units = CoverageUnitBuilder().build(artifact)
    by_id = {u.unit_id: u for u in units}
    retriever = CandidateRetriever(
        CoverageRetrievalConfig(min_retrieval_score=0.0),
        BagOfWordsEmbeddingBackend(),
    )

    checks = [
        (
            RequirementUnit(
                req_id="r-github",
                source_document_id="doc-tz",
                text="Исходный код всех частей сервиса должен храниться на веб-сервисе GitHub.",
                normalized_text="исходный код всех частей сервиса должен храниться на веб сервисе github",
                requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
            ),
            "4.9::sent2",
        ),
        (
            RequirementUnit(
                req_id="r-tech",
                source_document_id="doc-tz",
                text="Исходные коды программы должны быть написаны на языке программирования TypeScript с использованием библиотеки Angular.",
                normalized_text="исходные коды программы должны быть написаны на языке программирования typescript с использованием библиотеки angular",
                requirement_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
            ),
            "4.9::sent2",
        ),
        (
            RequirementUnit(
                req_id="r-rest",
                source_document_id="doc-tz",
                text="Все входные данные отправляются через REST API в формате JSON.",
                normalized_text="все входные данные отправляются через rest api в формате json",
                requirement_type=RequirementType.DATA_IO,
            ),
            "4.13::sent1",
        ),
        (
            RequirementUnit(
                req_id="r-figma",
                source_document_id="doc-tz",
                text="Макет интерфейса должен быть разработан в Figma.",
                normalized_text="макет интерфейса должен быть разработан в figma",
                requirement_type=RequirementType.INTERFACE,
            ),
            "4.16::sent6",
        ),
    ]

    for req, expected_fragment_id in checks:
        results = retriever.retrieve(req, units)
        top_fragment_ids = {by_id[r.unit_id].fragment_id for r in results[:3]}
        assert expected_fragment_id in top_fragment_ids


def test_polyakov_noise_fragments_are_demoted_for_pz_retrieval():
    cfg = CoverageRetrievalConfig(min_retrieval_score=0.0)
    retriever = CandidateRetriever(cfg, BagOfWordsEmbeddingBackend())
    req = RequirementUnit(
        req_id="r-ui",
        source_document_id="doc-tz",
        text="Клиентская часть должна представлять пользовательский интерфейс для просмотра проектов и загрузки файлов.",
        normalized_text="клиентская часть должна представлять пользовательский интерфейс для просмотра проектов и загрузки файлов",
        requirement_type=RequirementType.FUNCTIONAL,
    )
    correct = CoverageUnit(
        unit_id="u-correct",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Интерфейс веб-приложения позволяет просматривать проекты, открывать страницу проекта и загружать файлы.",
        normalized_text="интерфейс веб приложения позволяет просматривать проекты открывать страницу проекта и загружать файлы",
        metadata={"section_title": "Интерфейс веб-приложения"},
    )
    admin_noise = CoverageUnit(
        unit_id="u-admin",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Администратор – имеет возможность выдавать роли другим пользователям, создавать новые коллекции и редактировать их иерархию, удалять уже принятые исследования или добавлять новые, без необходимости проходить модерацию.",
        normalized_text="администратор имеет возможность выдавать роли другим пользователям создавать новые коллекции и редактировать их иерархию удалять уже принятые исследования или добавлять новые без необходимости проходить модерацию",
        metadata={"section_title": "Пользовательские сценарии"},
    )
    file_structure_noise = CoverageUnit(
        unit_id="u-files",
        target_document_id="doc-pz",
        target_doc_role="pz",
        text="Среди них: .ts, .html, .css, .scss, каждый из которых имеет одинаковое название и расположен в директории с таким же названием.",
        normalized_text="среди них ts html css scss каждый из которых имеет одинаковое название и расположен в директории с таким же названием",
        metadata={"section_title": "Обоснование средств разработки"},
    )

    results = retriever.retrieve(req, [admin_noise, file_structure_noise, correct])

    assert results[0].unit_id == "u-correct"
    scores = {r.unit_id: r.retrieval_score for r in results}
    assert scores["u-correct"] > scores["u-admin"]
    assert scores["u-correct"] > scores["u-files"]
