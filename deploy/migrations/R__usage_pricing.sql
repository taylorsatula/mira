-- store: postgres
-- notice: Makes sure MIRA's built-in price list has every entry this version expects. Prices you set yourself are kept.
INSERT INTO usage_pricing (name, input_price_per_mtok, output_price_per_mtok) VALUES
    ('__default__', 5.000000, 25.000000),
    ('primary', NULL, NULL),
    ('fast', NULL, NULL),
    ('batch', NULL, NULL),
    ('assessment', NULL, NULL),
    ('other', NULL, NULL)
ON CONFLICT (name) DO NOTHING;
