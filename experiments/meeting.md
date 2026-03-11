# Meeting Transcript
_2026-03-11 · 23m 30s · 3 speakers_

---

**unknown** `00:33 – 00:37`  
Hi there. How's it going? I'm good. How are you? I'm very good. Thank you.

**Kam** `00:37 – 00:41`  
Speaker 2: Awesome. So always, always seems to be a few of us play, always.

**Chris** `00:41 – 00:44`  
Speaker 1: No, it's fine to know if you're here or not. So I've dialed in in case.

**Kam** `00:45 – 00:53`  
2: We've got a workshop happening with GWS right now and it's just like running around and just like no. Chris is here. I want to have the face to face.

**Chris** `00:54 – 00:55`  
1: There we go. Thank you. You're odd.

**Kam** `00:56 – 01:17`  
2: Haha. Priorities. Priorities. Great. I just thought it would be useful for us to have a chat about what would be good, what good looks like in terms of content, in terms of engagement, in terms of things that would help folks. Because what I don't want to do is kind of stand there and talk at people for an hour.

**Chris** `01:17 – 01:20`  
1: I don't think that would be good.

**Kam** `01:20 – 01:23`  
2: And you know, stand there and go, well, yeah, they say hi.

**Chris** `01:25 – 01:30`  
1: Have I shown you the rest of the agenda so you've got the kind of context this is in? Do you want me to quickly? Whizzing.

**Kam** `01:30 – 01:35`  
2: Yeah, it would be helpful just to see where people are in.

**Chris** `01:35 – 01:54`  
1: That's right. So basically, first day is look back at 2025, go through. Brett will take us through the overall business strategy for this year. I'll take us through our overall IDG objectives for the year, but I will show you in a second just so you get a sense of those as well.

**Kam** `01:54 – 01:58`  
2: Because I'm only there on the two, the last two days, Wednesday, Thursday, yeah, yeah.

**Chris** `01:59 – 04:25`  
1: Yeah, so you won't be there for. This bit, and then most of that is we're doing some stuff with the learning development team on high-performing teams. Very good. And then a full morning the next morning on insights for leadership effectiveness. Yeah, so it's lots of people-y stuff, doing a bit of team building on the Tuesday afternoon. So then the Wednesday, so it's basically all analytics. So, all of the analytics business partners will go through their kind of key things they. Delivered and what they've got planned, and kind of a bit of foresight as to how they think it's going. Starting to think what we need to do for incorporating some of the newer non-beatrix brands into our core data sets. High-end data visualization, we've got our visualization expert coming on, he's awesome, really looking forward to that session. And then, this is the bit you're in, so Victor and Ben will do a What they're calling my first ML model, so getting everyone up to speed as to what it means by having the Marvin platform, what would it look like to deliver a model? What will we need to do? What are the inputs and outputs? So I think you'll find that interesting. And then you've got your hour, and then there's another hour on looking forward to between now and 2030. What are we not doing that we need to think about doing? So a little bit of visioning. And then day four is all platform stuff. So, how we scale up our data lake house using some of the new iSpoke technology and how we're going to structure data in there. We've now got, as well as the workshop doing data engineering, we've got Quantics. How do they play nicely? How do we give what to what? How do they use common standards and frameworks and tools? A forward-thinking bit on the data platform. Again, I think you'll have some good input into that, hopefully. And then Rob Smith, in terms of the infrastructure bit, how we plan and sequence that. And then a bit of visioning on the analytics platform itself. So Tableau and SuperSet and the other tools we've got. And then the other thing I thought was worth showing was our top-level OKRs for next year.

**Kam** `04:26 – 04:27`  
2: That would be helpful.

**Chris** `04:28 – 06:23`  
1: So Project 30. Consistent view of all data for all brands, whether they're matrix or not. That's a big statement, and there's a lot of work. It's easy to say. Yeah, a lot of work behind that. Basically, Marvin making it operational, functional, resilient. Actually, migrating stuff on the current platform, delivering new models, delivering them in a super resilient way so they can consistently deliver whether the warehouse is there or not. I'll talk about detail. DR, we're doing a big DR build out this year in a second data center for all our data products. So, what does that look like? The superset, so the operational reporting, lifting non-complex stuff off Tableau. Real-time alerting, we get lots of requests for alerting. We've never built too much off the data platform because it's not four nines resilient. But we want to introduce an element of real time. So that will be this year. Marketing analytics, we've got a lot of work to do to get more data and structure stuff differently and help them achieve their goals. And at the senior management off-site, we did something called Network No's, which I think you might have heard of this one. I've heard of it. Not seeing it. I can show you when we get a chance. But basically, we need to finish that bit of work. And those are my kind of top ones, and I'm not going to. Take all the others, but there's a shed load of cascaded ones as well that go into another level of detail. But don't worry about that. So, I guess kind of my asks for you for your session. You know, a lot of these will be collaborative, people feeding in, whatever. So that kind of style is good.

**unknown** `06:28 – 06:29`  
I can't believe that.

**Chris** `06:32 – 07:00`  
So, certainly, some of the stuff you're starting to do on harnessing AI in general in the org for productivity, what could that mean for this group? Anything you've got on kind of how AI is starting to change the data tooling landscape and how people think about warehousing and analytics and those kind of things. Don't know if you've got a think, but would for sure be. Interesting.

**Kam** `07:00 – 07:06`  
2: I mean, there's no end of hype. Yeah, well, that's kind of where my head is at the moment.

**Chris** `07:06 – 07:18`  
1: There was a company we were talking to that said, yeah, just give us all our data and law model it for you. Just like we curate this stuff very carefully. We have very specific metrics. Don't trust that.

**Kam** `07:18 – 08:32`  
2: There are definitely things that we just couldn't do before that you can do now. Especially around cleaning data, matching things. There was a load of stuff that was just Such a nightmare that actually the smart people right now is not even a thing anymore. And there's a whole bunch of not specifically around data warehousing, but there's a whole bunch of startups and companies that are now doing things which were completely unimaginable because the dirty data was just so dirty and complex. And you can have small language models clean things up for you and find things and aren't, you know, and so there are definitely things. That can happen on that. And I think, in terms of the, I will have a little thing, because there's definitely examples I could say, well, we couldn't do this, it would have taken an army of thousands of people, hundreds and hundreds of hours. Now you can just have a little server run over a weekend and do it for you. And it's because it needed some level of judgment, which LLM, even small language models have, but we could never write a piece of code that did that because it was impossible. There are definitely things that I can add there.

**Chris** `08:35 – 09:05`  
1: I think we do a lot of data ingestion and those kind of things. Is there interesting movements in that part of the world? I'm quite sure there are. Was there anything else top of mind? No, it's really kind of, as I think we've explained, as a group data group or wider group, people are kind of nervous to use tools at the moment because of the sensitivity of others. Data. Right. So, the work you're doing to try and unpack that and unpacking some of that will be really helpful.

**Kam** `09:05 – 09:37`  
2: Which is ready to go. Yeah. It's actually ready to go. I'm just, I'm just, it's, um, there's, there's a, there's a lot that we can, we can, we can leverage. It'll also be, and I'm just wondering to what degree, because ideally, it would be: I like the collaborative people feeding in, trying stuff out, so I'll have a think about what could be interactive, what. Could what could we run? I'm assuming everyone's going to have their laptops. I'm assuming we're all going to be on the VPN.

**Chris** `09:38 – 09:38`  
1: They all could be for sure.

**Kam** `09:39 – 10:04`  
2: So, you know, we're going to be attached to our internal networks. So, if there are things that we can do where actually people can maybe do something. Yeah, if they could touch it and feel it, they're going to be able to do it. Yes, that would be much better than just being told something because then you can walk away with having done something. Yeah. Yeah, because that's that you want to have an impact, and an impact. Well, unless there's someone really. And I'm not important, nor am I famous.

**Chris** `10:04 – 10:08`  
1: If you have someone important and I'm going to give you a big, you know, a big sound.

**Kam** `10:09 – 10:51`  
2: Oh, thanks. Thanks. Yeah. No, like, I don't know. Get Taylor Swift to come in and do a presentation. That'll have an impact. Don't need to touch anything. Thanks. Yeah. Okay. I've got that. All right. I think so. So I will think around that. I'll think what we can do putting things together because it's. Data ingestion you mentioned, some of those topics, I guess, some of the some of the kind of day-to-day work around, you know, and I haven't picked Rob on this, for instance, just like, oh, you know, I'm writing SQL things. I'm just like, you don't really need to do anything. I don't want to say it, like, but then there's.

**Chris** `10:51 – 11:02`  
1: Yeah, I mean, they do a lot of SQL tuning and those kind of things too. And just break the platform and, you know, getting. Assistance on those would save everybody loads of time.

**Kam** `11:02 – 11:14`  
2: Yeah, and it's the question of like what is sensitive. I think that's the other thing I haven't quite like, you know, are the of course PIIs are sensitive, but are the column names sensitive?

**Chris** `11:15 – 12:48`  
1: Are there, you know, what, you know, just figuring out what is yeah, you know, we obfuscate even column names when they're like, you know, use a password or to Security question or things like that, we obfuscate. Our current model is we secure by product, so you can see casino, sports, or poker, or any combination of the econ payments withdrawal stuff. We secure you by PII, so very, very few people get access to PII, and that's our depths in there. I think we also do something on people who can see financial data or non-financial data. So you can see numbers of players and numbers of games, but you can't see how many revenue numbers. So there's a few different elements that we secure the data on today. Makes sense. And I guess just in general, anything of you know, people will in that room will be relatively familiar with LLMs in concept at least. Not everyone will have in-depth knowledge, I for sure don't. You know, but where do you envisage things going over the next five years? This market is moving ludicrously fast, so anything you can give us that starts to unpack the future with your crystal ball out.

**Kam** `12:49 – 14:34`  
2: I read an article today of how even experts are getting the rate of improvement wrong. We're under, even the super forecast. Experts are getting the rate of improvement wrong. And the public is going faster than it is. It's going faster than the super forecasting experts. Yeah, interesting. Which was kind of quite a thing, and it contrasted it with what the general public think. And it's like the general public is thinking linearly. Like they're thinking, oh, it's here, then it'll be here. And experts. Yeah, or yeah, exactly. Or, you know, my job will change and I will do these other things at this point, which I think is always what's missed. I think there's still a lot of fear of like, you know, oh my god, it's going to do, it's going to take my job. The reality of it is, yes, it will take what you're doing, but that doesn't mean that you won't have employment, right? And it's that thing which really worries people. Yeah. I can certainly. I mean, if we're talking about the five years, there are some big macro changes that will be, that are already in play, but it's not really out there yet around particularly the foundational stuff, so the hardware stuff. The changes in hardware is, you know, I think what we're going to see next year, this year, is going to be a huge increase. In the cost of hardware. I'm thinking things will cost twice as much for the same thing.

**Chris** `14:36 – 14:38`  
1: Thinking specifically GPUs?

**Kam** `14:38 – 15:55`  
2: Yeah, anything with memory. Anything with memory and anything with GPUs. So, whereas maybe last year it cost $1,000, this year will cost $2,000. And I think it will probably be the following year before we start to see some of these new fabs kind of bringing things in that will then. Massively reduce the price of stuff, but this year is going to be one of oh shift. Like we've we budgeted for this, and if you want to buy that, it's why does it cost so much? I don't know if I'll go into the thing, but I think in the in the five-year, some of the things that we're gonna be seeing d definitely from a from a usage perspective, you know, and the role of I guess open open source. But I'm very keen to have folks try things. experiment with something, try something out play with something, maybe something that is a I'll think about that. So maybe I'll lamp that up. Something engaging. Because it's an entire, what I just don't want to do is stand there and say things that I just like, oh no, it's all going to get much better. And then nothing actually changes.

**unknown** `15:58 – 15:59`  
Where to find?

**Kam** `16:00 – 17:23`  
The other thing around opportunities for automation and kind of repetitive tasks, and maybe I could use a little bit of that time to try and and this is embryonic as a way of being able to capture some of those things beyond we just want this kind of you know this zero-based automation we might have chatted about rather you know to help folks understand Look, we've got this happening with this and this, but actually, these are the inputs, these are the outputs. You can rewrite them as a part rather than just building automations within like what we've got. And I'll try and make some examples of that in some ways. See where? Yeah, okay, that makes sense. I'm going to ask Brett if he can share with me the presentation just ahead of time if it's a bit earlier for the. Business strategy. I don't know if he's done it yet, but I don't know. He's got it yet. That one's going to be down to the wire for sure. But you're also producing. You've got this. This is the OKRs, and you have a present for the group that you're running. Security is going to be, like, Jaspel's going to be there as. No, okay, Jaspel's not here because I didn't see anything around that.

**Chris** `17:23 – 17:38`  
1: No, it's only all of my team, a couple of people from finance. One person from customer services, a few people from the workshop, a few people from Quantics, and you?

**Kam** `17:38 – 18:06`  
2: Yeah, okay. Totally nothing to do with this, but separately something that's kind of and I'm sure Victor's spoken to you around some kind of a validation around some of the ROI of these models and things like that, because the ROI is the other thing that we kind of need to be mindful of. And I'm wondering. Again, thinking through: hey, we've got AI, now we've got this massive bill. You know, what happened?

**Chris** `18:08 – 18:23`  
1: When I took Jack through these objectives, his immediate feedback was, yeah, great, Marvin, exactly what value did they do? Yes. He said, I know 100% supportive of it. He said, since targets, tell me what you're going to do. Yeah, so I have that challenge.

**Kam** `18:24 – 18:57`  
2: So I think that would be something in terms of validated, because it's one thing to say, well, We improve things, but it's another thing for kind of someone in finance to say, Yes, they did. You know, and this is we've lived, you know, because they're always they're always around, they're always writing checks. You know, how did it how did it improve stuff? Okay, so there'll be something in my head around how to do return on investment and then validated because that's generally how you end up getting more it actually worked.

**Chris** `18:57 – 18:59`  
1: Yeah, that'll that will play well.

**Kam** `18:59 – 19:11`  
2: Okay. So I'll just uh what else? Yeah, sounds. I'm gonna be thinking about this over the next well, it we've only got a couple of weeks, right?

**Chris** `19:12 – 19:16`  
1: Um yeah, it's a couple of full weeks after this week. I need to start writing my signs, I haven't started yet.

**Kam** `19:17 – 19:47`  
2: I'll stop filming something. I'll share some things with you in the week after. Um maybe next week, depending on this because I'm trying to get a the Transformation, AI transformation for software development in TWS at the moment. So just balance. I mean, you know, I wouldn't like to say, but you know, it's yeah, it's quite something.

**Chris** `19:48 – 19:48`  
1: Exactly.

**Kam** `19:51 – 21:54`  
2: He's brilliant, visionary, I think, you know, for a technical individual who's very visionary. He's not afraid. Quite different to a lot of the CTOs that I worked with. Quite like he's he's a di, he's, I don't know why they, you know, I know he studied like astrophysics and some things, but you know, he comes from, yeah, but he's CTOs are very rare. Well, this is the first CTO of a lot, you know, so and I'm thinking you have three or four hundred engineers, right? So I've worked with a few CTOs like that, and they are uber conservative, they don't want to break anything. They don't want to touch it. Like, the ones I've worked with. And Kai's like, I don't care if we throw out the whole thing. And I'm just like, what the hell? Like, you know, who is this? Like, because that's what a CEO normally says. Like, the CEO is like, ah, fuck us, we've got to do things differently. And usually the CTO is like, no, no, no, no, no, we can't get, you know, 10% improvement. Like, you know, we'll only just do this. So that's really nice. I think he's, yeah, that's the, I like that a lot. Yeah. It's definitely the right attitude. Well, we'll see how the rest of it goes. I think there's a lot of stuff that's happening in TWS that I kind of look at and go, Why are you doing all that? But there must be a reason. I just don't know what, I just don't have the context. So I assume that. So, yeah, so I'm just splitting between the industry, but I'm going to think about how do we divide this? How do we topics? What interactive things? Could be more experience things, try things out big tick in the box to me. Yeah, and just so hopefully, so hopefully, the impact would be: oh, we could do these things, we could do that this, and oh, I've tried this, and you know, rather than it, I tried it once and it doesn't work. Good. Let's, is there anything else? Is there anything else?

**Chris** `21:55 – 21:58`  
1: I don't think so. Okay. No, I don't want to spend too much of your time on this.

**Kam** `21:58 – 22:14`  
2: That's all right, wonderful. I'm looking forward to it. Yeah, yeah, likewise. I'll be arriving on the Tuesday evening. I think there's like, I think, do your own thing on Tuesday. I think. Please do your thing. Yeah, just settle in and then I'll ping you over one.

**unknown** `22:20 – 23:30`  
Thanks, sir. Thanks much appreciate it. See it. Should I leave it so? Yeah, I'll be in my game. Management.

