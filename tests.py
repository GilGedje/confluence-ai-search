from conf_serach_tool import Tools
import asyncio

async def noop_event_emitter(event):
    print(event)

tools = Tools()

async def main():
    result = await tools.search_confluence("what is ubuntu", "content", noop_event_emitter)
    print (result)

if __name__ == "__main__":
    asyncio.run(main())