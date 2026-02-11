# Integration Tests

## End-to-End Self-Reflection System Tests

### Purpose

These tests validate the complete workflow of the self-reflection system:
1. Trade logging
2. Performance metrics calculation
3. AI-powered reflection
4. Parameter adjustment (all 3 tiers)
5. Database consistency

### Test Coverage

- `test_complete_workflow_tier1`: Full workflow with auto-approval
- `test_complete_workflow_tier2_approved`: Workflow with Telegram approval
- `test_complete_workflow_tier3_rejected`: Workflow with emergency pause

### Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with output
pytest tests/integration/ -v -s

# Run specific test
pytest tests/integration/test_e2e_self_reflection.py::test_complete_workflow_tier1 -v
```

### Expected Results

All 3 tests should pass, demonstrating:
- ✅ Trades logged to database
- ✅ Outcomes tracked correctly
- ✅ Metrics calculated accurately
- ✅ Reflection generates insights
- ✅ Tier 1 adjustments auto-apply
- ✅ Tier 2 requests approval
- ✅ Tier 3 triggers emergency pause
- ✅ Database consistency maintained
