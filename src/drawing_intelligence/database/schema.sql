-- database/schema.sql
-- Updated schema with confidence constraints and optimized indexes

-- ============================================================
-- CORE TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS drawings (
    drawing_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL UNIQUE,
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pipeline_version TEXT,
    
    -- Overall confidence with validation
    overall_confidence REAL CHECK(overall_confidence BETWEEN 0.0 AND 1.0),
    
    -- Routing-specific confidence scores (from routing engine)
    routing_ocr_quality REAL CHECK(routing_ocr_quality BETWEEN 0.0 AND 1.0),
    routing_entity_completeness REAL CHECK(routing_entity_completeness BETWEEN 0.0 AND 1.0),
    routing_shape_quality REAL CHECK(routing_shape_quality BETWEEN 0.0 AND 1.0),
    routing_critical_field_presence REAL CHECK(routing_critical_field_presence BETWEEN 0.0 AND 1.0),
    routing_data_consistency REAL CHECK(routing_data_consistency BETWEEN 0.0 AND 1.0),
    
    -- Pipeline information
    pipeline_used TEXT CHECK(pipeline_used IN ('BASELINE', 'HYBRID', 'LLM_ENHANCED')),
    llm_stages_used TEXT,  -- JSON array of stages that used LLM
    
    needs_review BOOLEAN DEFAULT 0,
    review_flags TEXT,  -- JSON array of review flags
    
    component_hierarchy TEXT,  -- JSONB field for assembly hierarchy
    
    status TEXT CHECK(status IN ('processing', 'complete', 'failed', 'review')) DEFAULT 'processing'
);

CREATE TABLE IF NOT EXISTS text_extractions (
    text_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    content TEXT NOT NULL,
    
    -- Bounding box in pixels
    bbox_x INTEGER NOT NULL CHECK(bbox_x >= 0),
    bbox_y INTEGER NOT NULL CHECK(bbox_y >= 0),
    bbox_width INTEGER NOT NULL CHECK(bbox_width > 0),
    bbox_height INTEGER NOT NULL CHECK(bbox_height > 0),
    
    -- Confidence with OCR threshold validation
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
    low_confidence_flag BOOLEAN GENERATED ALWAYS AS (confidence < 0.85) STORED,
    
    ocr_engine TEXT CHECK(ocr_engine IN ('paddleocr', 'easyocr')),
    region_type TEXT CHECK(region_type IN ('dimension', 'label', 'note', 'title_block', 'table', 'text', 'figure'))
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    entity_type TEXT NOT NULL CHECK(entity_type IN (
        'PART_NUMBER', 'OEM', 'MATERIAL', 'DIMENSION', 
        'WEIGHT', 'THREAD_SPEC', 'TOLERANCE', 'SURFACE_FINISH',
        'QUANTITY', 'SCALE', 'REVISION', 'DATE', 'DRAWING_NUMBER'
    )),
    
    value TEXT NOT NULL,
    normalized_value TEXT,  -- JSON with normalized data
    original_text TEXT,     -- ✅ NEW: Original OCR text before normalization
    
    -- Confidence validation
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
    low_confidence_flag BOOLEAN GENERATED ALWAYS AS (confidence < 0.80) STORED,
    
    extraction_method TEXT CHECK(extraction_method IN ('regex', 'spacy', 'llm', 'llm_override')),
    
    -- Link to source text
    source_text_id TEXT REFERENCES text_extractions(text_id) ON DELETE SET NULL,
    
    -- Bounding box (inherited from source text, but stored for quick access)
    bbox_x INTEGER CHECK(bbox_x >= 0),
    bbox_y INTEGER CHECK(bbox_y >= 0),
    bbox_width INTEGER CHECK(bbox_width > 0),
    bbox_height INTEGER CHECK(bbox_height > 0)
);

CREATE TABLE IF NOT EXISTS shape_detections (
    detection_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    class_name TEXT NOT NULL,
    
    -- Confidence validation
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
    low_confidence_flag BOOLEAN GENERATED ALWAYS AS (confidence < 0.45) STORED,
    
    -- Bounding box
    bbox_x INTEGER NOT NULL CHECK(bbox_x >= 0),
    bbox_y INTEGER NOT NULL CHECK(bbox_y >= 0),
    bbox_width INTEGER NOT NULL CHECK(bbox_width > 0),
    bbox_height INTEGER NOT NULL CHECK(bbox_height > 0),
    
    -- Normalized bbox (0-1 scale for scale-independent storage)
    bbox_normalized_x REAL CHECK(bbox_normalized_x BETWEEN 0.0 AND 1.0),
    bbox_normalized_y REAL CHECK(bbox_normalized_y BETWEEN 0.0 AND 1.0),
    bbox_normalized_width REAL CHECK(bbox_normalized_width BETWEEN 0.0 AND 1.0),
    bbox_normalized_height REAL CHECK(bbox_normalized_height BETWEEN 0.0 AND 1.0),
    
    model_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS text_shape_associations (
    association_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    text_id TEXT NOT NULL REFERENCES text_extractions(text_id) ON DELETE CASCADE,
    detection_id TEXT NOT NULL REFERENCES shape_detections(detection_id) ON DELETE CASCADE,
    
    relationship_type TEXT CHECK(relationship_type IN ('dimension', 'label', 'note')),
    
    -- Confidence validation
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
    
    distance_pixels REAL CHECK(distance_pixels >= 0),
    
    -- Ensure unique associations
    UNIQUE(text_id, detection_id)
);

-- ============================================================
-- LLM TRACKING
-- ============================================================

CREATE TABLE IF NOT EXISTS llm_usage (
    usage_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    use_case TEXT NOT NULL CHECK(use_case IN (
        'drawing_assessment', 'ocr_verification', 
        'entity_extraction', 'shape_validation', 'complex_reasoning'
    )),
    
    provider TEXT NOT NULL CHECK(provider IN ('openai', 'anthropic', 'google')),
    model TEXT NOT NULL,  -- Full model ID (e.g., 'claude-3-sonnet-20240229')
    canonical_model_name TEXT,  -- Short name (e.g., 'claude-3-sonnet')
    model_tier INTEGER CHECK(model_tier IN (0, 1, 2)),  -- Tier at time of call
    
    tokens_input INTEGER NOT NULL CHECK(tokens_input >= 0),
    tokens_output INTEGER NOT NULL CHECK(tokens_output >= 0),
    image_count INTEGER DEFAULT 0 CHECK(image_count >= 0),
    
    cost_usd REAL NOT NULL CHECK(cost_usd >= 0),
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Was this a step-down from original model choice?
    tier_stepped_down BOOLEAN DEFAULT 0
);

-- ============================================================
-- AUDIT TRAIL
-- ============================================================

CREATE TABLE IF NOT EXISTS processing_audit (
    audit_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    stage TEXT NOT NULL CHECK(stage IN (
        'pdf_processing', 'image_preprocessing', 'ocr_extraction',
        'entity_extraction', 'shape_detection', 'data_association',
        'quality_scoring', 'llm_enhancement', 'export'
    )),
    
    status TEXT NOT NULL CHECK(status IN ('started', 'completed', 'failed', 'skipped')),
    
    duration_seconds REAL CHECK(duration_seconds >= 0),
    error_message TEXT,
    error_traceback TEXT,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional metadata
    metadata TEXT  -- JSON with stage-specific details
);

-- ============================================================
-- VALIDATION TRACKING
-- ============================================================

CREATE TABLE IF NOT EXISTS validation_issues (
    issue_id TEXT PRIMARY KEY,
    drawing_id TEXT NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
    
    severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    issue_type TEXT NOT NULL,
    message TEXT NOT NULL,
    
    -- References to problematic entities/detections
    entity_id TEXT REFERENCES entities(entity_id) ON DELETE SET NULL,
    detection_id TEXT REFERENCES shape_detections(detection_id) ON DELETE SET NULL,
    text_id TEXT REFERENCES text_extractions(text_id) ON DELETE SET NULL,
    
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Drawing queries
CREATE INDEX IF NOT EXISTS idx_drawings_status ON drawings(status);
CREATE INDEX IF NOT EXISTS idx_drawings_review ON drawings(needs_review);
CREATE INDEX IF NOT EXISTS idx_drawings_status_confidence 
    ON drawings(status, overall_confidence);
CREATE INDEX IF NOT EXISTS idx_drawings_review_timestamp 
    ON drawings(needs_review, processing_timestamp);

-- Entity queries
CREATE INDEX IF NOT EXISTS idx_entities_drawing ON entities(drawing_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_drawing_type ON entities(drawing_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_low_conf ON entities(low_confidence_flag) 
    WHERE low_confidence_flag = 1;
CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_text_id);

-- Shape detection queries
CREATE INDEX IF NOT EXISTS idx_detections_drawing ON shape_detections(drawing_id);
CREATE INDEX IF NOT EXISTS idx_detections_class ON shape_detections(class_name);
CREATE INDEX IF NOT EXISTS idx_detections_drawing_class 
    ON shape_detections(drawing_id, class_name);
CREATE INDEX IF NOT EXISTS idx_detections_low_conf ON shape_detections(low_confidence_flag)
    WHERE low_confidence_flag = 1;

-- Text extraction queries
CREATE INDEX IF NOT EXISTS idx_text_drawing ON text_extractions(drawing_id);
CREATE INDEX IF NOT EXISTS idx_text_region ON text_extractions(region_type);
CREATE INDEX IF NOT EXISTS idx_text_low_conf ON text_extractions(low_confidence_flag)
    WHERE low_confidence_flag = 1;

-- Association queries
CREATE INDEX IF NOT EXISTS idx_associations_drawing ON text_shape_associations(drawing_id);
CREATE INDEX IF NOT EXISTS idx_associations_text ON text_shape_associations(text_id);
CREATE INDEX IF NOT EXISTS idx_associations_shape ON text_shape_associations(detection_id);

-- LLM usage queries
CREATE INDEX IF NOT EXISTS idx_llm_drawing ON llm_usage(drawing_id);
CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_use_case ON llm_usage(use_case);
CREATE INDEX IF NOT EXISTS idx_llm_stepped_down ON llm_usage(tier_stepped_down)
    WHERE tier_stepped_down = 1;

-- Audit queries
CREATE INDEX IF NOT EXISTS idx_audit_drawing ON processing_audit(drawing_id);
CREATE INDEX IF NOT EXISTS idx_audit_stage ON processing_audit(stage);
CREATE INDEX IF NOT EXISTS idx_audit_status ON processing_audit(status);
CREATE INDEX IF NOT EXISTS idx_audit_drawing_stage ON processing_audit(drawing_id, stage);

-- Validation queries
CREATE INDEX IF NOT EXISTS idx_validation_drawing ON validation_issues(drawing_id);
CREATE INDEX IF NOT EXISTS idx_validation_severity ON validation_issues(severity);
CREATE INDEX IF NOT EXISTS idx_validation_drawing_severity 
    ON validation_issues(drawing_id, severity);

-- ============================================================
-- FULL-TEXT SEARCH
-- ============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    entity_id UNINDEXED,
    entity_type,
    value,
    original_text,  -- ✅ NEW: Also index original text
    content=entities,
    content_rowid=rowid
);

-- Triggers to keep FTS index updated
CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(entity_id, entity_type, value, original_text)
    VALUES (new.entity_id, new.entity_type, new.value, new.original_text);
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON entities BEGIN
    DELETE FROM entities_fts WHERE entity_id = old.entity_id;
END;

CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON entities BEGIN
    UPDATE entities_fts 
    SET entity_type = new.entity_type, 
        value = new.value,
        original_text = new.original_text
    WHERE entity_id = old.entity_id;
END;

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- View: Drawings needing review with reasons
CREATE VIEW IF NOT EXISTS drawings_needing_review AS
SELECT 
    d.drawing_id,
    d.source_file,
    d.overall_confidence,
    d.pipeline_used,
    d.review_flags,
    COUNT(DISTINCT vi.issue_id) as issue_count,
    MAX(vi.severity) as highest_severity,
    d.processing_timestamp
FROM drawings d
LEFT JOIN validation_issues vi ON d.drawing_id = vi.drawing_id
WHERE d.needs_review = 1
GROUP BY d.drawing_id
ORDER BY 
    CASE MAX(vi.severity)
        WHEN 'CRITICAL' THEN 4
        WHEN 'HIGH' THEN 3
        WHEN 'MEDIUM' THEN 2
        WHEN 'LOW' THEN 1
        ELSE 0
    END DESC,
    d.processing_timestamp DESC;

-- View: Daily LLM cost summary
CREATE VIEW IF NOT EXISTS llm_daily_costs AS
SELECT 
    DATE(timestamp) as date,
    use_case,
    canonical_model_name,
    COUNT(*) as call_count,
    SUM(tokens_input) as total_input_tokens,
    SUM(tokens_output) as total_output_tokens,
    SUM(image_count) as total_images,
    SUM(cost_usd) as total_cost,
    AVG(cost_usd) as avg_cost_per_call,
    SUM(CASE WHEN tier_stepped_down = 1 THEN 1 ELSE 0 END) as stepped_down_count
FROM llm_usage
GROUP BY DATE(timestamp), use_case, canonical_model_name
ORDER BY date DESC, total_cost DESC;

-- View: Processing success rate
CREATE VIEW IF NOT EXISTS processing_statistics AS
SELECT 
    DATE(processing_timestamp) as date,
    COUNT(*) as total_drawings,
    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    SUM(CASE WHEN needs_review = 1 THEN 1 ELSE 0 END) as needing_review,
    AVG(overall_confidence) as avg_confidence,
    SUM(CASE WHEN pipeline_used = 'LLM_ENHANCED' THEN 1 ELSE 0 END) as llm_enhanced_count
FROM drawings
GROUP BY DATE(processing_timestamp)
ORDER BY date DESC;

-- ============================================================
-- EXAMPLE QUERIES
-- ============================================================

-- Find drawings with missing PART_NUMBER
-- SELECT d.drawing_id, d.source_file
-- FROM drawings d
-- WHERE NOT EXISTS (
--     SELECT 1 FROM entities e 
--     WHERE e.drawing_id = d.drawing_id 
--     AND e.entity_type = 'PART_NUMBER'
--     AND e.confidence > 0.70
-- );

-- Find low confidence OCR that should have triggered fallback
-- SELECT * FROM text_extractions
-- WHERE low_confidence_flag = 1
-- AND ocr_engine = 'paddleocr';

-- Daily LLM cost breakdown
-- SELECT * FROM llm_daily_costs
-- WHERE date = CURRENT_DATE;

-- Drawings with validation issues
-- SELECT * FROM drawings_needing_review
-- WHERE highest_severity IN ('HIGH', 'CRITICAL')
-- LIMIT 10;