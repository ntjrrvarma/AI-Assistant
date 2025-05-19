from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation
import asyncio
import pyttsx3
import ollama

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

    async def process_input(self, text: str) -> str:
        try:
            response = ollama.chat(
                model='llama3:8b-instruct',
                messages=[{'role': 'user', 'content': self.instructions + "\n\n" + text}]
            )
            return response['message']['content']
        except ollama.OllamaAPIError as e:
            print(f"Ollama API Error: {e}")
            return "Sorry, I encountered an issue."

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession()
    agent = Assistant()
    session.agent = agent

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

    engine = pyttsx3.init()
    greeting = "Hello! How can I assist you today?"
    print(f"Agent says: {greeting}")
    engine.say(greeting)
    engine.runAndWait()

    user_spoken_future = asyncio.Future()

    async def user_spoke_listener():
        try:
            # Try potential ways to access incoming events
            async for event in session._get_incoming_events(): # Placeholder - replace with actual method
                if isinstance(event, agents.UserSpoke) and not user_spoken_future.done():
                    user_spoken_future.set_result(True)
                elif isinstance(event, agents.UserDisconnected):
                    print(f"User {event.user_id} disconnected.")
                    if not user_spoken_future.done():
                        user_spoken_future.cancel()
                    break
                elif isinstance(event, agents.SessionEnded):
                    print("Session ended.")
                    if not user_spoken_future.done():
                        user_spoken_future.cancel()
                    break
        except Exception as e:
            print(f"Error in user_spoke_listener: {e}")
            if not user_spoken_future.done():
                user_spoken_future.set_exception(e)

    listener_task = asyncio.create_task(user_spoke_listener())

    try:
        await asyncio.wait_for(user_spoken_future, timeout=1)
        print("User started speaking within 1 second.")
    except asyncio.TimeoutError:
        print("No immediate response from the user.")
    except asyncio.CancelledError:
        print("Waiting for user spoken was cancelled.")

    try:
        # Replace this with the correct way to iterate over incoming events
        async def process_events():
            async for event in session._get_incoming_events(): # Placeholder - replace with actual method
                if isinstance(event, agents.UserSpoke):
                    user_audio = event.audio
                    transcript = await session.transcribe(user_audio)
                    print(f"User said: {transcript}")
                    llm_response_text = await agent.process_input(transcript)
                    print(f"LLM Response: {llm_response_text}")
                    engine.say(llm_response_text)
                    engine.runAndWait()
                    follow_up_response = await session.generate_reply(
                        instructions="Ask the user if they need anything else."
                    )
                    print(f"Follow-up LLM Response: {follow_up_response}")
                    engine.say(follow_up_response)
                    engine.runAndWait()
                elif isinstance(event, agents.UserDisconnected):
                    print(f"User {event.user_id} disconnected.")
                    break
                elif isinstance(event, agents.SessionEnded):
                    print("Session ended.")
                    break
        await process_events()

    except Exception as e:
        print(f"An error occurred in the main event loop: {e}")
    finally:
        if not listener_task.done():
            listener_task.cancel()
            await asyncio.gather(listener_task, return_exceptions=True)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))