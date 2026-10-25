CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    company_name VARCHAR(120) NOT NULL UNIQUE,
    registered_at DATE NOT NULL,
    employee_count INTEGER NOT NULL CHECK (employee_count > 0),
    segment VARCHAR(32) NOT NULL CHECK (
        segment IN (
            'micro_business',
            'small_business',
            'mid_market',
            'enterprise'
        )
    ),
    region VARCHAR(64) NOT NULL,
    acquisition_channel VARCHAR(32) NOT NULL CHECK (
        acquisition_channel IN (
            'organic',
            'paid_ads',
            'partner',
            'referral',
            'outbound'
        )
    ),
    CHECK (
        (employee_count BETWEEN 1 AND 10
            AND segment = 'micro_business')
        OR
        (employee_count BETWEEN 11 AND 50
            AND segment = 'small_business')
        OR
        (employee_count BETWEEN 51 AND 250
            AND segment = 'mid_market')
        OR
        (employee_count > 250
            AND segment = 'enterprise')
    )
);

CREATE TABLE subscriptions (
    subscription_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL
        REFERENCES customers(customer_id)
        ON DELETE RESTRICT,
    plan VARCHAR(16) NOT NULL CHECK (
        plan IN ('basic', 'pro', 'enterprise')
    ),
    started_at DATE NOT NULL,
    ended_at DATE,
    status VARCHAR(16) NOT NULL CHECK (
        status IN ('active', 'cancelled')
    ),
    licensed_users INTEGER NOT NULL CHECK (licensed_users > 0),
    billing_cycle VARCHAR(16) NOT NULL CHECK (
        billing_cycle IN ('monthly', 'annual')
    ),
    cancellation_reason VARCHAR(32) CHECK (
        cancellation_reason IN (
            'too_expensive',
            'missing_features',
            'technical_issues',
            'switched_to_competitor',
            'no_longer_needed'
        )
    ),
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK (
        (status = 'active'
            AND ended_at IS NULL
            AND cancellation_reason IS NULL)
        OR
        (status = 'cancelled'
            AND ended_at IS NOT NULL
            AND cancellation_reason IS NOT NULL)
    )
);

CREATE TABLE payments (
    payment_id BIGINT PRIMARY KEY,
    subscription_id BIGINT NOT NULL
        REFERENCES subscriptions(subscription_id)
        ON DELETE RESTRICT,
    paid_at DATE NOT NULL,
    amount_rub NUMERIC(12, 2) NOT NULL CHECK (amount_rub > 0),
    operation_type VARCHAR(16) NOT NULL CHECK (
        operation_type IN ('charge', 'refund')
    ),
    status VARCHAR(16) NOT NULL CHECK (
        status IN ('succeeded', 'failed')
    ),
    billing_period_start DATE NOT NULL,
    billing_period_end DATE NOT NULL,
    CHECK (billing_period_end >= billing_period_start)
);

CREATE TABLE support_tickets (
    ticket_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL
        REFERENCES customers(customer_id)
        ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    category VARCHAR(16) NOT NULL CHECK (
        category IN (
            'billing',
            'technical',
            'plan',
            'access',
            'other'
        )
    ),
    priority VARCHAR(16) NOT NULL CHECK (
        priority IN ('low', 'normal', 'high')
    ),
    status VARCHAR(16) NOT NULL CHECK (
        status IN ('open', 'closed')
    ),
    channel VARCHAR(16) NOT NULL CHECK (
        channel IN ('chat', 'email', 'web')
    ),
    CHECK (closed_at IS NULL OR closed_at >= created_at),
    CHECK (
        (status = 'open' AND closed_at IS NULL)
        OR
        (status = 'closed' AND closed_at IS NOT NULL)
    )
);

CREATE INDEX idx_customers_segment
    ON customers(segment);

CREATE INDEX idx_customers_registered_at
    ON customers(registered_at);

CREATE INDEX idx_subscriptions_customer_id
    ON subscriptions(customer_id);

CREATE INDEX idx_subscriptions_plan
    ON subscriptions(plan);

CREATE INDEX idx_subscriptions_started_at
    ON subscriptions(started_at);

CREATE INDEX idx_subscriptions_ended_at
    ON subscriptions(ended_at);

CREATE INDEX idx_payments_subscription_id
    ON payments(subscription_id);

CREATE INDEX idx_payments_paid_at
    ON payments(paid_at);

CREATE INDEX idx_support_tickets_customer_id
    ON support_tickets(customer_id);

CREATE INDEX idx_support_tickets_created_at
    ON support_tickets(created_at);

CREATE INDEX idx_support_tickets_category
    ON support_tickets(category);
