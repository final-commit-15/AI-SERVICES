-- AgentForge PostgreSQL Initialization Script
-- This script runs on first database creation

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agentforge;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS auth;

-- Set search path
ALTER DATABASE agentforge SET search_path TO agentforge, analytics, auth, public;

-- Users table
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    roles TEXT[] DEFAULT ARRAY['user'],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Keys table
CREATE TABLE IF NOT EXISTS auth.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    prefix VARCHAR(8) NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    roles TEXT[] DEFAULT ARRAY['user'],
    scopes TEXT[] DEFAULT ARRAY[]::TEXT[],
    rate_limit INTEGER,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

-- Conversations table
CREATE TABLE IF NOT EXISTS agentforge.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    title VARCHAR(500),
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    message_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS agentforge.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES agentforge.conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON agentforge.messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON agentforge.messages(role);

-- Usage Records table (Analytics)
CREATE TABLE IF NOT EXISTS analytics.usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES agentforge.conversations(id) ON DELETE SET NULL,
    provider VARCHAR(100) NOT NULL,
    model VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    status VARCHAR(50) NOT NULL,
    error TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for usage records
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON analytics.usage_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_user ON analytics.usage_records(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON analytics.usage_records(provider);
CREATE INDEX IF NOT EXISTS idx_usage_model ON analytics.usage_records(model);
CREATE INDEX IF NOT EXISTS idx_usage_task_type ON analytics.usage_records(task_type);
CREATE INDEX IF NOT EXISTS idx_usage_conversation ON analytics.usage_records(conversation_id);

-- Provider Health Logs
CREATE TABLE IF NOT EXISTS analytics.provider_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider VARCHAR(100) NOT NULL,
    model VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    latency_ms REAL,
    error TEXT,
    details JSONB DEFAULT '{}'
);

-- Cost Alerts
CREATE TABLE IF NOT EXISTS analytics.cost_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threshold_usd REAL NOT NULL,
    current_spend_usd REAL NOT NULL,
    period VARCHAR(50) NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID REFERENCES auth.users(id) ON DELETE SET NULL
);

-- Vector embeddings table (for pgvector if used directly)
CREATE TABLE IF NOT EXISTS agentforge.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON agentforge.embeddings USING hnsw (embedding vector_cosine_ops);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON auth.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON agentforge.conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agentforge TO agentforge;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO agentforge;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO agentforge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agentforge TO agentforge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO agentforge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA auth TO agentforge;