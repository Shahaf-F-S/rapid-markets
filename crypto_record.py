
from dataclasses import dataclass, field

from rapid_markets.source import Feed


@dataclass(frozen=True, slots=True)
class FeedsManager:

    type FeedID = str

    feeds: dict[FeedID, Feed] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.values())

    def keys(self) -> tuple[FeedID, ...]:
        return tuple(self.feeds.keys())

    def values(self) -> tuple[Feed, ...]:
        return tuple(self.feeds.values())

    @staticmethod
    def id_number(feed_id: FeedID) -> int:
        return int(feed_id.split('-', maxsplit=1)[0])

    @staticmethod
    def id_exchange(feed_id: FeedID) -> str:
        return feed_id.split('-', maxsplit=0)[0]

    def exchange_feeds(self, exchange: str) -> set[Feed]:
        return set(feed for feed in self.feeds.values() if feed.name == exchange)

    def exchange_id_numbers(self, exchange: str) -> set[int]:
        return set(self.id_number(uid) for uid, feed in self.feeds.items() if feed.name == exchange)

    def exchange_ids(self, exchange: str) -> set[str]:
        return set(uid for uid, feed in self.feeds.items() if feed.name == exchange)

    def new_id(self, exchange: str) -> FeedID:
        uid = max(self.exchange_id_numbers(exchange) or (0,)) + 1
        return f'{exchange}-{uid}'

    def new(self, exchange: str) -> tuple[FeedID, Feed]:
        feed_id = self.new_id(exchange)
        feed = Feed(exchange)
        self.feeds[feed_id] = feed
        return feed_id, feed

    def get(self, feed_id: FeedID) -> Feed:
        return self.feeds[feed_id]
