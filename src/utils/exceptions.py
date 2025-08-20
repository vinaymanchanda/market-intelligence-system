class MarketIntelligenceError(Exception):
    pass


class ScrapingError(MarketIntelligenceError):
    pass


class RateLimitError(MarketIntelligenceError):
    pass


class DataProcessingError(MarketIntelligenceError):
    pass


class StorageError(MarketIntelligenceError):
    pass


class AnalysisError(MarketIntelligenceError):
    pass


class ConfigurationError(MarketIntelligenceError):
    pass


class AuthenticationError(MarketIntelligenceError):
    pass


class ValidationError(MarketIntelligenceError):
    pass
